import { reviewCandidate } from "./auto_review.js";
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const WIKI_API = "https://ko.wikipedia.org/w/api.php";
const USER_AGENT = "QuantumWikiResearch/1.0 (https://github.com/DEV1ND/BC-COUNTRY-)";
const MAX_RESULTS = 8;
const SYNC_BATCH = 5;
const MAX_PAGE_CHARS = 80000;

function json(data,status=200){return new Response(JSON.stringify(data,null,2),{status,headers:JSON_HEADERS});}
async function sha256(value){
  const bytes=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value));
  return [...new Uint8Array(bytes)].map(x=>x.toString(16).padStart(2,"0")).join("");
}
async function wikiRequest(params){
  const url=new URL(WIKI_API);
  url.search=new URLSearchParams({...params,format:"json",formatversion:"2",maxlag:"5"});
  const res=await fetch(url,{headers:{"user-agent":USER_AGENT,"accept":"application/json"},signal:AbortSignal.timeout(15000)});
  if(res.status===429||res.status===503) throw new Error("Wikimedia rate limit/server busy; retry on a later scheduled run");
  if(!res.ok) throw new Error("Wikipedia API HTTP "+res.status);
  const data=await res.json();
  if(data.error) throw new Error(data.error.info||data.error.code);
  return data;
}
async function collectTopic(topic,env){
  const search=await wikiRequest({action:"query",list:"search",srsearch:topic.query,srnamespace:"0",srlimit:String(MAX_RESULTS),srprop:"snippet|timestamp"});
  let discovered=0,updated=0;
  for(const hit of search.query?.search||[]){
    const detail=await wikiRequest({action:"query",prop:"extracts|info",explaintext:"1",exsectionformat:"plain",inprop:"url",pageids:String(hit.pageid)});
    const page=detail.query?.pages?.[0], text=String(page?.extract||"").trim().slice(0,MAX_PAGE_CHARS);
    if(!page||text.length<500) continue;
    const hash=await sha256(text);
    const sameHash=await env.DB.prepare("SELECT id,url FROM pages WHERE content_hash=?").bind(hash).first();
    const sourceUrl=page.fullurl||("https://ko.wikipedia.org/wiki/"+encodeURIComponent(page.title.replace(/ /g,"_")));
    if(sameHash&&sameHash.url!==sourceUrl) continue;
    const existing=await env.DB.prepare("SELECT id,content_hash FROM pages WHERE url=?").bind(sourceUrl).first();
    if(existing&&existing.content_hash===hash){
      await env.DB.prepare("UPDATE pages SET last_seen_at=datetime('now') WHERE id=?").bind(existing.id).run();
      continue;
    }
    const id=existing?.id||crypto.randomUUID();
    if(existing){
      await env.DB.prepare(`UPDATE pages SET title=?,snippet=?,content_hash=?,content=?,revision_id=?,status='pending_review',
        license_status='attribution_required',license_name='CC BY-SA 4.0',
        attribution='Wikipedia contributors; retain page title, URL, license, and indicate modifications',
        discovered_at=datetime('now'),last_seen_at=datetime('now'),reviewed_at=NULL,reviewer_note=NULL,synced_at=NULL,sync_error=NULL
        WHERE id=?`).bind(page.title,String(hit.snippet||"").replace(/<[^>]+>/g,""),hash,text,page.lastrevid||null,id).run();
      updated++;
    }else{
      await env.DB.prepare(`INSERT INTO pages
        (id,provider,title,url,domain,snippet,content_hash,content,r2_key,status,license_status,license_name,attribution,revision_id,discovered_at,last_seen_at)
        VALUES (?,'wikipedia',?,?,?,?,?,?, '', 'pending_review','attribution_required','CC BY-SA 4.0',
        'Wikipedia contributors; retain page title, URL, license, and indicate modifications',?,datetime('now'),datetime('now'))`)
        .bind(id,page.title,sourceUrl,"ko.wikipedia.org",String(hit.snippet||"").replace(/<[^>]+>/g,""),hash,text,page.lastrevid||null).run();
      discovered++;
    }
  }
  await env.DB.prepare("UPDATE search_topics SET last_searched_at=datetime('now'),last_error=NULL WHERE query=?").bind(topic.query).run();
  return {discovered,updated};
}

async function autoReviewPending(env, limit = 15) {
  const { results = [] } = await env.DB.prepare("SELECT id,title,url,domain,content,license_status,license_name,attribution,revision_id FROM pages WHERE status='pending_review' AND reviewer_note IS NULL ORDER BY discovered_at LIMIT ?").bind(limit).all();
  const counts = { approved: 0, excluded: 0, pending: 0, checked: results.length };
  for (const row of results) {
    const decision = reviewCandidate(row);
    const note = "AUTO_REVIEW: " + decision.reason + " | score=" + decision.score;
    await env.DB.prepare("UPDATE pages SET status=?,reviewed_at=CASE WHEN ?='approved' OR ?='excluded' THEN datetime('now') ELSE reviewed_at END,reviewer_note=? WHERE id=? AND status='pending_review'")
      .bind(decision.status, decision.status, decision.status, note, row.id).run();
    if (decision.status === "approved") counts.approved++;
    else if (decision.status === "excluded") counts.excluded++;
    else counts.pending++;
  }
  return counts;
}

async function syncApproved(env){
  if(!env.QUANTUM_INFERENCE_URL||!env.QUANTUM_API_TOKEN) return {ok:false,synced:0,error:"inference_sync_not_configured"};
  const headers={"authorization":"Bearer "+env.QUANTUM_API_TOKEN};
  let cacheState;
  try{
    const statusResponse=await fetch(new URL("/admin/knowledge/ids",env.QUANTUM_INFERENCE_URL),{
      headers,signal:AbortSignal.timeout(15000)
    });
    cacheState=await statusResponse.json().catch(()=>({}));
    if(!statusResponse.ok || !Array.isArray(cacheState.page_ids))
      return {ok:false,synced:0,error:"inference_ids_rejected",status:statusResponse.status,detail:cacheState};
  }catch(e){return {ok:false,synced:0,error:"inference_ids_unreachable",detail:String(e.message||e)};}
  const cachedIds=new Set(cacheState.page_ids.map(String));
  const expected=await env.DB.prepare("SELECT COUNT(*) AS count FROM pages WHERE status='approved' AND length(content)>=200").first();
  const candidates=await env.DB.prepare("SELECT id FROM pages WHERE status='approved' AND length(content)>=200 ORDER BY id").all();
  const missing=(candidates.results||[]).filter(row=>!cachedIds.has(String(row.id))).slice(0,SYNC_BATCH);
  if(!missing.length) return {ok:true,synced:0,cacheDocuments:cachedIds.size,expectedDocuments:expected?.count||0,rebuilding:false};
  const placeholders=missing.map(()=>"?").join(",");
  const rowsResult=await env.DB.prepare("SELECT id,title,url,content,license_name,attribution FROM pages WHERE id IN ("+placeholders+") AND status='approved'").bind(...missing.map(row=>row.id)).all();
  const rows=rowsResult.results||[];
  if(!rows.length) return {ok:true,synced:0,cacheDocuments:cachedIds.size,expectedDocuments:expected?.count||0,rebuilding:cachedIds.size<(expected?.count||0),warning:"missing_ids_not_found"};
  const docs=rows.map(row=>({page_id:row.id,title:row.title,url:row.url,text:String(row.content||"").trim(),license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors"}));
  let response;
  try{
    response=await fetch(new URL("/admin/knowledge/import",env.QUANTUM_INFERENCE_URL),{
      method:"POST",headers:{"content-type":"application/json",...headers},
      body:JSON.stringify({documents:docs}),signal:AbortSignal.timeout(30000)
    });
  }catch(e){return {ok:false,synced:0,error:"inference_unreachable",detail:String(e.message||e)};}
  const importResult=await response.json().catch(()=>({}));
  if(!response.ok) return {ok:false,synced:0,error:"inference_rejected",status:response.status,detail:importResult};
  for(const doc of docs) await env.DB.prepare("UPDATE pages SET synced_at=datetime('now'),sync_error=NULL WHERE id=?").bind(doc.page_id).run();
  return {ok:true,synced:docs.length,inference:importResult,cacheDocumentsBefore:cachedIds.size,expectedDocuments:expected?.count||0,cacheDocumentsAfter:cachedIds.size+docs.length};
}
async function maybeStartIncrementalTraining(env){
  if(!env.QUANTUM_INFERENCE_URL||!env.QUANTUM_API_TOKEN)
    return {started:false,reason:"inference_training_not_configured"};
  const count=await env.DB.prepare("SELECT COUNT(*) AS count FROM pages WHERE status='approved' AND synced_at IS NOT NULL").first();
  if((count?.count||0)<10) return {started:false,reason:"need_at_least_10_approved_synced_pages",documents:count?.count||0};
  const previous=await env.DB.prepare("SELECT value FROM settings WHERE key='training_last_requested_at'").first();
  if(previous?.value){
    const due=await env.DB.prepare("SELECT CASE WHEN datetime(?) <= datetime('now','-1 day') THEN 1 ELSE 0 END AS due").bind(previous.value).first();
    if(!due?.due) return {started:false,reason:"daily_training_interval_not_reached",documents:count.count};
  }
  let response;
  try{
    response=await fetch(new URL("/admin/training/start",env.QUANTUM_INFERENCE_URL),{
      method:"POST",headers:{"authorization":"Bearer "+env.QUANTUM_API_TOKEN},
      signal:AbortSignal.timeout(15000)
    });
  }catch(e){return {started:false,reason:"training_service_unreachable",detail:String(e.message||e)};}
  const result=await response.json().catch(()=>({}));
  if(!response.ok) return {started:false,reason:"training_service_rejected",status:response.status,detail:result};
  if(result.started){
    await env.DB.prepare("INSERT INTO settings(key,value) VALUES('training_last_requested_at',datetime('now')) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run();
  }
  return {...result,documents:count.count};
}
async function runCycle(env){
  const state=await env.DB.prepare("SELECT value FROM settings WHERE key='running'").first();
  if(!state||state.value!=="1") return {running:false,discovered:0,updated:0,synced:0,message:"paused"};
  const {results:topics=[]}=await env.DB.prepare("SELECT query FROM search_topics WHERE enabled=1 ORDER BY last_searched_at IS NOT NULL,last_searched_at LIMIT 2").all();
  let discovered=0,updated=0; const errors=[];
  for(const topic of topics){
    try{const r=await collectTopic(topic,env);discovered+=r.discovered;updated+=r.updated;}
    catch(e){
      const message=String(e.message||e); errors.push({topic:topic.query,error:message});
      await env.DB.prepare("UPDATE search_topics SET last_error=? WHERE query=?").bind(message.slice(0,500),topic.query).run();
    }
  }
  const review=await autoReviewPending(env,15);
  const sync=await syncApproved(env);
  const training={started:false,reason:"manual_candidate_training_only",automatic_training_disabled:true};
  await env.DB.prepare("INSERT INTO crawl_runs(discovered,updated,synced,errors,ran_at) VALUES(?,?,?,?,datetime('now'))")
    .bind(discovered,updated,sync.synced||0,JSON.stringify({crawl:errors,sync:sync.error||null})).run();
  return {running:true,provider:"ko.wikipedia.org",topics:topics.length,discovered,updated,review,sync,training,errors};
}
async function approvePage(request,env){
  let body;try{body=await request.json();}catch{return json({error:"invalid_json"},400);}
  const id=String(body.id||"").trim(),note=String(body.note||"").trim().slice(0,1000);
  if(!id)return json({error:"id_required"},400);
  const page=await env.DB.prepare("SELECT id,status,license_name FROM pages WHERE id=?").bind(id).first();
  if(!page)return json({error:"page_not_found"},404);
  if(page.status!=="pending_review")return json({error:"only_pending_pages_can_be_approved",status:page.status},409);
  await env.DB.prepare("UPDATE pages SET status='approved',reviewed_at=datetime('now'),reviewer_note=? WHERE id=?").bind(note,id).run();
  return json({ok:true,id,status:"approved",license:page.license_name,notice:"CC BY-SA attribution and share-alike obligations still apply."});
}
function authorized(request,env){return !!env.ADMIN_TOKEN&&request.headers.get("authorization")==="Bearer "+env.ADMIN_TOKEN;}
export default {
  async fetch(request,env){
    const url=new URL(request.url);
    if(request.method==="GET"&&url.pathname==="/health")return json({ok:true,service:"quantum-wikipedia-explorer",isolated:true,provider:"ko.wikipedia.org"});
    if(!authorized(request,env))return json({error:"unauthorized"},401);
    if(request.method==="GET"&&url.pathname==="/training-status"){
      if(!env.QUANTUM_INFERENCE_URL||!env.QUANTUM_API_TOKEN)return json({error:"training_service_not_configured"},503);
      try{
        const response=await fetch(new URL("/admin/training/status",env.QUANTUM_INFERENCE_URL),{
          headers:{"authorization":"Bearer "+env.QUANTUM_API_TOKEN},signal:AbortSignal.timeout(15000)
        });
        const result=await response.json().catch(()=>({}));
        return json(result,response.status);
      }catch(e){return json({error:"training_service_unreachable",detail:String(e.message||e)},502);}
    }
    if(request.method==="GET"&&url.pathname==="/status"){
      const running=await env.DB.prepare("SELECT value FROM settings WHERE key='running'").first();
      const counts=await env.DB.prepare("SELECT status,COUNT(*) AS count FROM pages GROUP BY status").all();
      const pendingSync=await env.DB.prepare("SELECT COUNT(*) AS count FROM pages WHERE status='approved' AND synced_at IS NULL").first();
      const latest=await env.DB.prepare("SELECT discovered,updated,synced,errors,ran_at FROM crawl_runs ORDER BY id DESC LIMIT 1").first();
      const topics=await env.DB.prepare("SELECT COUNT(*) AS total,SUM(enabled) AS enabled FROM search_topics").first();
      return json({running:running?.value==="1",provider:"ko.wikipedia.org",pageCounts:counts.results||[],pendingSync:pendingSync?.count||0,topics,latestRun:latest||null});
    }
    if(request.method==="GET"&&url.pathname==="/pages"){
      const status=String(url.searchParams.get("status")||"pending_review");
      const limit=Math.min(50,Math.max(1,Number(url.searchParams.get("limit")||20)));
      const cursor=String(url.searchParams.get("before")||"");
      const rows=await env.DB.prepare("SELECT id,title,url,snippet,status,license_name,revision_id,discovered_at,last_seen_at,synced_at FROM pages WHERE status=? AND (?='' OR discovered_at<?) ORDER BY discovered_at DESC LIMIT ?").bind(status,cursor,cursor,limit).all();
      return json({pages:rows.results||[]});
    }
    if(request.method==="GET"&&url.pathname==="/export-approved"){
      const limit=Math.min(10,Math.max(1,Number(url.searchParams.get("limit")||5)));
      const after=String(url.searchParams.get("after")||"");
      const rows=await env.DB.prepare("SELECT id,title,url,content,license_name,attribution FROM pages WHERE status='approved' AND id>? ORDER BY id LIMIT ?").bind(after,limit).all();
      const documents=(rows.results||[]).map(row=>({
        page_id:row.id,title:row.title,url:row.url,text:String(row.content||""),
        license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors"
      })).filter(row=>row.text.length>=200);
      return json({documents,next_after:rows.results?.length?rows.results[rows.results.length-1].id:null,has_more:(rows.results||[]).length===limit});
    }
    if(request.method==="GET"&&url.pathname==="/topics"){
      const rows=await env.DB.prepare("SELECT query,enabled,last_searched_at,last_error FROM search_topics ORDER BY query").all();
      return json({topics:rows.results||[]});
    }
    if(request.method==="POST"&&url.pathname==="/topics"){
      let body;try{body=await request.json();}catch{return json({error:"invalid_json"},400);}
      const query=String(body.query||"").trim().slice(0,160);
      if(query.length<2)return json({error:"query_too_short"},400);
      await env.DB.prepare("INSERT INTO search_topics(query,enabled) VALUES(?,1) ON CONFLICT(query) DO UPDATE SET enabled=1").bind(query).run();
      return json({ok:true,query});
    }
    if(request.method==="POST"&&url.pathname==="/start"){
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','1') ON CONFLICT(key) DO UPDATE SET value='1'").run();
      return json({ok:true,running:true});
    }
    if(request.method==="POST"&&url.pathname==="/pause"){
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','0') ON CONFLICT(key) DO UPDATE SET value='0'").run();
      return json({ok:true,running:false});
    }
    if(request.method==="POST"&&url.pathname==="/run-once")return json(await runCycle(env));
    if(request.method==="POST"&&url.pathname==="/auto-review"){let body={};try{body=await request.json();}catch{} const limit=Math.min(50,Math.max(1,Number(body.limit||20))); return json({ok:true,review:await autoReviewPending(env,limit)});}
    if(request.method==="POST"&&url.pathname==="/approve")return approvePage(request,env);
    if(request.method==="POST"&&url.pathname==="/sync-approved")return json(await syncApproved(env));
    return json({error:"not_found"},404);
  },
  async scheduled(_event,env,ctx){ctx.waitUntil(runCycle(env));}
};
