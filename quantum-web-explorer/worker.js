const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const WIKI_API = "https://ko.wikipedia.org/w/api.php";
const USER_AGENT = "QuantumWikiResearch/1.0 (configure a public contact URL or email)";
const MAX_RESULTS = 10;
const SYNC_BATCH = 10;

function json(data, status=200) {
  return new Response(JSON.stringify(data,null,2),{status,headers:JSON_HEADERS});
}
async function sha256(text) {
  const digest=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,"0")).join("");
}
async function wikiRequest(params) {
  const url=new URL(WIKI_API);
  url.search=new URLSearchParams({...params,format:"json",formatversion:"2",maxlag:"5"});
  const res=await fetch(url,{headers:{"user-agent":USER_AGENT,"accept":"application/json"},signal:AbortSignal.timeout(15000)});
  if(res.status===429||res.status===503) {
    const wait=Number(res.headers.get("retry-after")||5);
    throw new Error("Wikimedia throttled; retry after "+wait+" seconds");
  }
  if(!res.ok) throw new Error("Wikipedia API HTTP "+res.status);
  const data=await res.json();
  if(data.error) throw new Error(data.error.info||data.error.code);
  return data;
}
async function collectTopic(topic,env) {
  const search=await wikiRequest({
    action:"query",list:"search",srsearch:topic.query,srnamespace:"0",
    srlimit:String(MAX_RESULTS),srprop:"snippet|timestamp"
  });
  let discovered=0;
  for(const hit of (search.query?.search||[])) {
    const detail=await wikiRequest({
      action:"query",prop:"extracts|info",explaintext:"1",exsectionformat:"plain",
      inprop:"url",pageids:String(hit.pageid)
    });
    const page=detail.query?.pages?.[0];
    const text=String(page?.extract||"").trim();
    if(!page||text.length<400) continue;
    const hash=await sha256(text);
    const exists=await env.DB.prepare("SELECT id FROM pages WHERE content_hash=?").bind(hash).first();
    if(exists) continue;
    const id=crypto.randomUUID();
    const key="pages/"+id+".txt";
    const sourceUrl=page.fullurl||("https://ko.wikipedia.org/wiki/"+encodeURIComponent(page.title.replace(/ /g,"_")));
    await env.RAW_PAGES.put(key,text,{httpMetadata:{contentType:"text/plain; charset=utf-8"}});
    await env.DB.prepare(
      "INSERT INTO pages (id,provider,title,url,domain,snippet,content_hash,r2_key,status,license_status,license_name,attribution,discovered_at) VALUES (?,'wikipedia',?,?,?,?,?,?, 'pending_review','attribution_required','CC BY-SA 4.0','Wikipedia contributors; retain page title, URL, license, and indicate modifications',datetime('now'))"
    ).bind(id,page.title,sourceUrl,"ko.wikipedia.org",String(hit.snippet||"").replace(/<[^>]+>/g,""),hash,key).run();
    discovered++;
  }
  await env.DB.prepare("UPDATE search_topics SET last_searched_at=datetime('now') WHERE query=?").bind(topic.query).run();
  return discovered;
}
async function runCycle(env) {
  const state=await env.DB.prepare("SELECT value FROM settings WHERE key='running'").first();
  if(!state||state.value!=="1") return {running:false,discovered:0,message:"paused"};
  const {results:topics=[]}=await env.DB.prepare(
    "SELECT query FROM search_topics WHERE enabled=1 ORDER BY last_searched_at IS NOT NULL,last_searched_at LIMIT 2"
  ).all();
  let discovered=0; const errors=[];
  // Sequential requests, low concurrency, and maxlag protect Wikimedia infrastructure.
  for(const topic of topics) {
    try { discovered+=await collectTopic(topic,env); }
    catch(e) { errors.push(String(e.message||e)); }
  }
  await env.DB.prepare("INSERT INTO crawl_runs(discovered,errors,ran_at) VALUES(?,?,datetime('now'))")
    .bind(discovered,JSON.stringify(errors.slice(0,5))).run();
  return {running:true,provider:"ko.wikipedia.org",topics:topics.length,discovered,errors};
}

async function approvePage(request,env) {
  let body; try { body=await request.json(); } catch { return json({error:"invalid_json"},400); }
  const id=String(body.id||"").trim(), note=String(body.note||"").trim().slice(0,1000);
  if(!id) return json({error:"id_required"},400);
  const page=await env.DB.prepare("SELECT id,status,license_name FROM pages WHERE id=?").bind(id).first();
  if(!page) return json({error:"page_not_found"},404);
  if(page.status!=="pending_review") return json({error:"only_pending_pages_can_be_approved",status:page.status},409);
  await env.DB.prepare("UPDATE pages SET status='approved',reviewed_at=datetime('now'),reviewer_note=? WHERE id=?").bind(note,id).run();
  return json({ok:true,id,status:"approved",license:page.license_name,notice:"CC BY-SA attribution and share-alike obligations still apply."});
}
async function syncApproved(env) {
  if(!env.QUANTUM_INFERENCE_URL||!env.QUANTUM_API_TOKEN) return json({error:"inference_sync_not_configured"},503);
  const {results=[]}=await env.DB.prepare("SELECT id,title,url,r2_key,license_name,attribution FROM pages WHERE status='approved' AND synced_at IS NULL ORDER BY reviewed_at LIMIT ?").bind(SYNC_BATCH).all();
  if(!results.length) return json({ok:true,synced:0,message:"no_approved_documents_waiting"});
  const docs=[];
  for(const row of results) {
    const obj=await env.RAW_PAGES.get(row.r2_key); if(!obj) continue;
    const text=await obj.text(); if(!text.trim()) continue;
    docs.push({page_id:row.id,title:row.title,url:row.url,text:text.slice(0,120000),license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors"});
  }
  if(!docs.length) return json({ok:false,synced:0,error:"approved_r2_documents_missing"},502);
  let response;
  try {
    response=await fetch(new URL("/admin/knowledge/import",env.QUANTUM_INFERENCE_URL),{
      method:"POST",headers:{"content-type":"application/json","authorization":"Bearer "+env.QUANTUM_API_TOKEN},
      body:JSON.stringify({documents:docs}),signal:AbortSignal.timeout(30000)
    });
  } catch(e) { return json({ok:false,synced:0,error:"inference_unreachable",detail:String(e.message||e)},502); }
  const result=await response.json().catch(()=>({}));
  if(!response.ok) return json({ok:false,synced:0,error:"inference_rejected",status:response.status,detail:result},502);
  for(const doc of docs) await env.DB.prepare("UPDATE pages SET synced_at=datetime('now'),sync_error=NULL WHERE id=?").bind(doc.page_id).run();
  return json({ok:true,synced:docs.length,remaining:Math.max(0,results.length-docs.length),inference:result});
}

function authorized(request,env) {
  return !!env.ADMIN_TOKEN&&request.headers.get("authorization")==="Bearer "+env.ADMIN_TOKEN;
}
export default {
  async fetch(request,env) {
    const url=new URL(request.url);
    if(request.method==="GET"&&url.pathname==="/health")
      return json({ok:true,service:"quantum-wikipedia-explorer",isolated:true,provider:"ko.wikipedia.org"});
    if(!authorized(request,env)) return json({error:"unauthorized"},401);
    if(request.method==="GET"&&url.pathname==="/status") {
      const running=await env.DB.prepare("SELECT value FROM settings WHERE key='running'").first();
      const counts=await env.DB.prepare("SELECT status,COUNT(*) AS count FROM pages GROUP BY status").all();
      const pendingSync=await env.DB.prepare("SELECT COUNT(*) AS count FROM pages WHERE status='approved' AND synced_at IS NULL").first();
      const latest=await env.DB.prepare("SELECT discovered,ran_at FROM crawl_runs ORDER BY id DESC LIMIT 1").first();
      return json({running:running?.value==="1",provider:"ko.wikipedia.org",pageCounts:counts.results||[],pendingSync:pendingSync?.count||0,latestRun:latest||null});
    }
    if(request.method==="POST"&&url.pathname==="/start") {
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','1') ON CONFLICT(key) DO UPDATE SET value='1'").run();
      return json({ok:true,running:true});
    }
    if(request.method==="POST"&&url.pathname==="/pause") {
      await env.DB.prepare("INSERT INTO settings(key,value) VALUES('running','0') ON CONFLICT(key) DO UPDATE SET value='0'").run();
      return json({ok:true,running:false});
    }
    if(request.method==="POST"&&url.pathname==="/run-once") return json(await runCycle(env));
    if(request.method==="POST"&&url.pathname==="/approve") return approvePage(request,env);
    if(request.method==="POST"&&url.pathname==="/sync-approved") return syncApproved(env);
    return json({error:"not_found"},404);
  },
  async scheduled(_event,env,ctx) { ctx.waitUntil(runCycle(env)); }
};