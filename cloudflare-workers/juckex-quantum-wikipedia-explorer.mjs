import { reviewCandidate, validateManualApproval } from "./auto_review.mjs";
import { planMissingPageIds, isCandidateInferenceTarget, buildSyncPreview } from "./knowledge_sync_plan.mjs";
import { verifyCloudflareAccessIdentity, makeVerifiedReviewNote, parseVerifiedReviewNote } from "./access_identity.mjs";
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const WIKI_API = "https://ko.wikipedia.org/w/api.php";
const USER_AGENT = "QuantumWikiResearch/1.0 (https://github.com/DEV1ND/BC-COUNTRY-)";
const MAX_RESULTS = 8;
const SYNC_BATCH = 5;
const MAX_PAGE_CHARS = 50000;
const REVIEW_HTML = `<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>JUCKEX Quantum · Knowledge Review</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#131722;--line:#293244;--gold:#d7b66a;--muted:#9aa6b8;--text:#edf1f7;--red:#d98282}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#24202a 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 18px 60px}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}
.brand{font-size:12px;letter-spacing:.18em;color:var(--gold);font-weight:700}.heading{font-size:25px;font-weight:750;margin:4px 0}.sub{color:var(--muted);font-size:13px}
.card{background:linear-gradient(145deg,#171d29,#10141d);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:16px;box-shadow:0 14px 40px #0004}
input,textarea{width:100%;background:#0b1018;border:1px solid #303a4d;color:var(--text);border-radius:10px;padding:12px;font:inherit}
textarea{min-height:105px;resize:vertical}button{border:1px solid #76633b;background:#302817;color:#f5dfad;border-radius:10px;padding:11px 15px;font:inherit;font-weight:650;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:320px minmax(0,1fr);gap:16px}
#queue{display:grid;gap:8px;max-height:70vh;overflow:auto}.item{width:100%;text-align:left;background:#101620;border-color:#2b3547;color:var(--text);font-weight:500}.item small{display:block;color:var(--muted);font-size:11px}
#article{white-space:pre-wrap;overflow-wrap:anywhere;max-height:55vh;overflow:auto;background:#0b1018;border:1px solid #273144;border-radius:10px;padding:15px;font-size:13px}
a{color:#e7c77f;overflow-wrap:anywhere}.meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.field{display:grid;gap:7px;margin:13px 0}.label{font-size:12px;color:#c4cddd;font-weight:650}
.notice{color:#e8c985;background:#2a2418;border:1px solid #655333;padding:12px;border-radius:10px;font-size:13px}
#error{color:#ffaaaa;min-height:22px;font-size:13px}#detail[hidden],#login[hidden]{display:none}
@media(max-width:820px){.grid{grid-template-columns:1fr}#queue{max-height:240px}.top{align-items:flex-start;flex-direction:column}main{padding:20px 12px 40px}}
</style>
</head>
<body><main>
<header class="top"><div><div class="brand">JUCKEX · QUANTUM</div><div class="heading">지식 문서 검토 센터</div><div class="sub">위키백과 문서는 독립적인 출처와 대조한 뒤에만 승인할 수 있습니다.</div></div><div class="meta">Candidate 전용 · 모델 가중치 변경 없음</div></header>
<section id="login" class="card">
 <div class="label">관리자 토큰</div><div class="row"><input id="token" type="password" autocomplete="off" placeholder="Cloudflare ADMIN_TOKEN"><button id="connect">연결</button></div>
 <p class="meta">토큰은 이 페이지가 열려 있는 동안 메모리에만 보관됩니다. 승인 작업은 Cloudflare Access로 인증된 허용 목록 검토자만 할 수 있습니다.</p>
</section>
<div id="error" role="status"></div>
<section id="workspace" class="grid" hidden>
 <aside class="card"><div class="row" style="justify-content:space-between"><strong>검토 대기 문서</strong><button id="refresh">새로고침</button></div><div id="queue" style="margin-top:14px"></div></aside>
 <section id="detail" class="card" hidden>
  <h2 id="title" style="margin:0 0 8px;font-size:21px"></h2>
  <div class="meta">원문: <a id="source" target="_blank" rel="noopener noreferrer"></a></div>
  <div class="meta" id="provenance" style="margin:9px 0 15px"></div>
  <div id="article"></div>
  <div class="notice" style="margin-top:14px">승인 전 확인: 문서의 핵심 주장과 수치·날짜를 독립적인 신뢰 출처와 대조하세요. 자동 점수는 사실 정확도를 보증하지 않습니다.</div>
  <div class="field"><label class="label" for="summary">사실 검토 요약</label><textarea id="summary" placeholder="어떤 핵심 내용을 어떤 근거로 확인했는지 적어주세요."></textarea></div>
  <div class="field"><label class="label" for="reference">독립적인 HTTPS 대조 출처</label><input id="reference" type="url" placeholder="https://..."></div>
  <div class="row"><button id="approve">검토 완료 및 승인</button><button id="skip" style="background:#182131;border-color:#344158;color:#d7e0ef">보류</button></div>
 </section>
</section>
</main>
<script>
let adminToken="",selected=null;
const el=id=>document.getElementById(id);
function showError(message){el("error").textContent=message||"";}
async function api(path,options={}){
 const headers=Object.assign({},options.headers||{},{authorization:"Bearer "+adminToken});
 const response=await fetch(path,Object.assign({},options,{headers}));
 const data=await response.json().catch(()=>({}));
 if(!response.ok)throw new Error(data.detail||data.error||("HTTP "+response.status));
 return data;
}
async function connect(){
 adminToken=el("token").value.trim();
 if(!adminToken){showError("관리자 토큰을 입력하세요.");return;}
 try{await api("/status");el("login").hidden=true;el("workspace").hidden=false;showError("");await loadQueue();}
 catch(error){adminToken="";showError("연결 실패: "+error.message);}
}
async function loadQueue(){
 try{
  const data=await api("/pages?status=pending_review&limit=50");
  const queue=el("queue");queue.replaceChildren();
  for(const page of data.pages||[]){
   const button=document.createElement("button");button.className="item";
   button.textContent=page.title||"(제목 없음)";
   const small=document.createElement("small");small.textContent="리비전 "+(page.revision_id||"없음")+" · "+(page.discovered_at||"");
   button.appendChild(small);button.addEventListener("click",()=>openPage(page.id));queue.appendChild(button);
  }
  if(!(data.pages||[]).length){const p=document.createElement("p");p.className="meta";p.textContent="검토 대기 문서가 없습니다.";queue.appendChild(p);}
  showError("");
 }catch(error){showError("목록 조회 실패: "+error.message);}
}
async function openPage(id){
 try{
  selected=await api("/page?id="+encodeURIComponent(id));
  el("title").textContent=selected.title;
  el("source").href=selected.url;el("source").textContent=selected.url;
  el("provenance").textContent="리비전: "+selected.revision_id+" · SHA-256: "+selected.content_hash+" · 라이선스: "+selected.license_name;
  el("article").textContent=selected.content;
  el("summary").value="";el("reference").value="";el("detail").hidden=false;showError("");
 }catch(error){showError("문서 조회 실패: "+error.message);}
}
async function approve(){
 if(!selected)return;
 const summary=el("summary").value.trim(),reference=el("reference").value.trim();
 try{
  const parsed=new URL(reference);
  if(parsed.protocol!=="https:"||parsed.hostname.endsWith("wikipedia.org"))throw new Error("위키백과 외부 HTTPS 출처를 입력하세요.");
  if(summary.length<20)throw new Error("사실 검토 요약을 20자 이상 작성하세요.");
  await api("/approve",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({id:selected.id,note:summary,reference_url:reference})});
  showError("승인 완료: "+selected.title);selected=null;el("detail").hidden=true;await loadQueue();
 }catch(error){showError("승인할 수 없습니다: "+error.message);}
}
el("connect").addEventListener("click",connect);
el("token").addEventListener("keydown",event=>{if(event.key==="Enter")connect();});
el("refresh").addEventListener("click",loadQueue);
el("approve").addEventListener("click",approve);
el("skip").addEventListener("click",()=>{selected=null;el("detail").hidden=true;});
</script></body></html>`;

function json(data,status=200){return new Response(JSON.stringify(data,null,2),{status,headers:JSON_HEADERS});}
function buildWikipediaAttribution(title,url){
  return `Wikipedia contributors, ${title}, ${url}; licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Modifications must be indicated.`;
}
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
    const detail=await wikiRequest({action:"query",prop:"extracts|info|revisions",explaintext:"1",exsectionformat:"plain",inprop:"url",rvprop:"ids",rvlimit:"1",pageids:String(hit.pageid)});
    const page=detail.query?.pages?.[0], text=String(page?.extract||"").trim().slice(0,MAX_PAGE_CHARS);
    if(!page||text.length<500) continue;
    const hash=await sha256(text);
    const sameHash=await env.DB.prepare("SELECT id,url FROM pages WHERE content_hash=?").bind(hash).first();
    const sourceUrl=page.fullurl||("https://ko.wikipedia.org/wiki/"+encodeURIComponent(page.title.replace(/ /g,"_")));
    if(sameHash&&sameHash.url!==sourceUrl) continue;
    const attribution=buildWikipediaAttribution(page.title,sourceUrl);
    const existing=await env.DB.prepare("SELECT id,content_hash FROM pages WHERE url=?").bind(sourceUrl).first();
    if(existing&&existing.content_hash===hash){
      await env.DB.prepare("UPDATE pages SET last_seen_at=datetime('now') WHERE id=?").bind(existing.id).run();
      continue;
    }
    const id=existing?.id||crypto.randomUUID();
    if(existing){
      await env.DB.prepare(`UPDATE pages SET title=?,snippet=?,content_hash=?,content=?,revision_id=?,status='pending_review',
        license_status='attribution_required',license_name='CC BY-SA 4.0',attribution=?,
        discovered_at=datetime('now'),last_seen_at=datetime('now'),reviewed_at=NULL,reviewer_note=NULL,synced_at=NULL,sync_error=NULL
        WHERE id=?`).bind(page.title,String(hit.snippet||"").replace(/<[^>]+>/g,""),hash,text,attribution,page.lastrevid||null,id).run();
      updated++;
    }else{
      await env.DB.prepare(`INSERT INTO pages
        (id,provider,title,url,domain,snippet,content_hash,content,r2_key,status,license_status,license_name,attribution,revision_id,discovered_at,last_seen_at)
        VALUES (?,'wikipedia',?,?,?,?,?,?, '', 'pending_review','attribution_required','CC BY-SA 4.0',?,?,datetime('now'),datetime('now'))`)
        .bind(id,page.title,sourceUrl,"ko.wikipedia.org",String(hit.snippet||"").replace(/<[^>]+>/g,""),hash,text,attribution,page.lastrevid||null).run();
      discovered++;
    }
  }
  await env.DB.prepare("UPDATE search_topics SET last_searched_at=datetime('now'),last_error=NULL WHERE query=?").bind(topic.query).run();
  return {discovered,updated};
}

async function autoReviewPending(env, limit = 15) {
  const { results = [] } = await env.DB.prepare("SELECT id,provider,title,url,domain,content,content_hash,license_status,license_name,attribution,revision_id FROM pages WHERE status='pending_review' AND reviewer_note IS NULL ORDER BY discovered_at LIMIT ?").bind(limit).all();
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
  if(!env.QUANTUM_API_TOKEN || !isCandidateInferenceTarget(env.QUANTUM_INFERENCE_URL))
    return {ok:false,synced:0,error:"candidate_inference_target_required"};
  const headers={"authorization":"Bearer "+env.QUANTUM_API_TOKEN};
  let cacheStatus;
  try{
    const statusResponse=await fetch(new URL("/admin/knowledge/status",env.QUANTUM_INFERENCE_URL),{
      headers,signal:AbortSignal.timeout(15000)
    });
    cacheStatus=await statusResponse.json().catch(()=>({}));
    if(!statusResponse.ok) return {ok:false,synced:0,error:"inference_status_rejected",status:statusResponse.status,detail:cacheStatus};
  }catch(e){return {ok:false,synced:0,error:"inference_status_unreachable",detail:String(e.message||e)};}
  if(!Array.isArray(cacheStatus.page_ids) || cacheStatus.page_ids_complete !== true)
    return {ok:false,synced:0,error:"inference_status_missing_complete_page_ids",detail:"Deploy the compatible inference status endpoint before enabling ID-based reconciliation."};
  const cachedIds=new Set(cacheStatus.page_ids.map(value=>String(value)));
  const expected=await env.DB.prepare("SELECT COUNT(*) AS count FROM pages WHERE status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%' AND length(content)>=200").first();
  const approved=await env.DB.prepare("SELECT id FROM pages WHERE status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%' AND length(content)>=200 ORDER BY id").all();
  const approvedIds=(approved.results||[]).map(row=>String(row.id));
  const missingIds=planMissingPageIds(approvedIds,cachedIds,SYNC_BATCH);
  const rebuilding=cachedIds.size<Number(expected?.count||0);
  if(!missingIds.length) {
    return {ok:true,synced:0,cacheDocuments:cachedIds.size,expectedDocuments:expected?.count||0,rebuilding:false};
  }
  const placeholders=missingIds.map(()=>"?").join(",");
  const result=await env.DB.prepare("SELECT id,title,url,content,content_hash,revision_id,license_name,attribution FROM pages WHERE id IN ("+placeholders+") AND status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%' ORDER BY id").bind(...missingIds).all();
  const rows=result.results||[];
  if(rows.length!==missingIds.length)
    return {ok:false,synced:0,error:"approved_document_batch_changed_during_sync",expectedBatch:missingIds.length,selected:rows.length};
  const docs=rows.map(row=>({page_id:row.id,title:row.title,url:row.url,text:String(row.content||"").trim(),license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors",revision_id:row.revision_id,content_hash:row.content_hash}));
  let response;
  try{
    response=await fetch(new URL("/admin/knowledge/import",env.QUANTUM_INFERENCE_URL),{
      method:"POST",headers:{"content-type":"application/json",...headers},
      body:JSON.stringify({documents:docs}),signal:AbortSignal.timeout(30000)
    });
  }catch(e){return {ok:false,synced:0,error:"inference_unreachable",detail:String(e.message||e)};}
  const importResult=await response.json().catch(()=>({}));
  if(!response.ok) return {ok:false,synced:0,error:"inference_rejected",status:response.status,detail:importResult};
  if(Number(importResult.imported)!==docs.length)
    return {ok:false,synced:0,error:"inference_import_count_mismatch",expected:docs.length,received:importResult.imported??null};
  let marked=0;
  for(const doc of docs) {
    const update=await env.DB.prepare("UPDATE pages SET synced_at=datetime('now'),sync_error=NULL WHERE id=? AND status='approved'").bind(doc.page_id).run();
    if(Number(update.meta?.changes||0)!==1)
      return {ok:false,synced:marked,error:"d1_sync_marker_mismatch",page_id:doc.page_id,changes:update.meta?.changes??null};
    marked++;
  }
  return {ok:true,synced:marked,inference:importResult,rebuilding,cacheDocumentsBefore:cachedIds.size,expectedDocuments:expected?.count||0};
}
async function previewApprovedSync(env) {
  if (!env.QUANTUM_API_TOKEN || !isCandidateInferenceTarget(env.QUANTUM_INFERENCE_URL))
    return {ok:false,dryRun:true,error:"candidate_inference_target_required",writes_performed:false};
  let cacheStatus;
  try {
    const response = await fetch(new URL("/admin/knowledge/status", env.QUANTUM_INFERENCE_URL), {
      headers: {"authorization":"Bearer "+env.QUANTUM_API_TOKEN},
      signal: AbortSignal.timeout(15000)
    });
    cacheStatus = await response.json().catch(() => ({}));
    if (!response.ok)
      return {ok:false,dryRun:true,error:"inference_status_rejected",status:response.status,writes_performed:false};
  } catch (error) {
    return {ok:false,dryRun:true,error:"inference_status_unreachable",detail:String(error.message||error),writes_performed:false};
  }
  if (!Array.isArray(cacheStatus.page_ids) || cacheStatus.page_ids_complete !== true)
    return {ok:false,dryRun:true,error:"inference_status_missing_complete_page_ids",writes_performed:false};
  const approved = await env.DB.prepare(
    "SELECT id FROM pages WHERE status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%' AND length(content)>=200 ORDER BY id"
  ).all();
  const approvedIds = (approved.results||[]).map(row => String(row.id));
  const preview = buildSyncPreview(approvedIds, cacheStatus.page_ids, SYNC_BATCH);
  return {
    ok:true,
    dryRun:true,
    ...preview,
    candidateKnowledgeDocuments:Number(cacheStatus.documents||0),
    candidateIndexHealthy:cacheStatus.integrity?.healthy===true,
    writes_performed:false,
    note:"D1과 Candidate 데이터는 변경하지 않았습니다. 실제 동기화는 별도 POST /sync-approved 요청에서만 실행됩니다."
  };
}

async function maybeStartIncrementalTraining(){
  return {started:false,reason:"automatic_training_disabled",note:"Wikipedia retrieval updates never modify Quantum model weights."};
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
  const training=await maybeStartIncrementalTraining(env);
  await env.DB.prepare("INSERT INTO crawl_runs(discovered,updated,synced,errors,ran_at) VALUES(?,?,?,?,datetime('now'))")
    .bind(discovered,updated,sync.synced||0,JSON.stringify({crawl:errors,sync:sync.error||null})).run();
  return {running:true,provider:"ko.wikipedia.org",topics:topics.length,discovered,updated,review,sync,training,errors};
}
async function approvePage(request,env,identity){
  let body;try{body=await request.json();}catch{return json({error:"invalid_json"},400);}
  const id=String(body.id||"").trim();
  const note=String(body.note||"").trim().slice(0,1000);
  const referenceUrl=String(body.reference_url||"").trim().slice(0,2000);
  if(!id)return json({error:"id_required"},400);
  let reference;
  try{reference=new URL(referenceUrl);}catch{return json({error:"independent_reference_required"},400);}
  if(reference.protocol!=="https:"||reference.hostname==="ko.wikipedia.org"||reference.hostname.endsWith(".wikipedia.org"))
    return json({error:"independent_reference_required"},400);
  const page=await env.DB.prepare("SELECT id,status,provider,title,url,domain,content,content_hash,license_status,license_name,attribution,revision_id FROM pages WHERE id=?").bind(id).first();
  if(!page)return json({error:"page_not_found"},404);
  if(page.status!=="pending_review")return json({error:"only_pending_pages_can_be_approved",status:page.status},409);
  const factCheckNote=note+" | 독립 대조 출처: "+referenceUrl;
  const validation=await validateManualApproval(page,factCheckNote);
  if(!validation.ok)return json({error:"manual_review_required",detail:validation.reason},422);
  let reviewerNote;
  try{reviewerNote=makeVerifiedReviewNote(note,referenceUrl,identity);}
  catch{return json({error:"verified_reviewer_identity_required"},403);}
  const update=await env.DB.prepare("UPDATE pages SET status='approved',reviewed_at=datetime('now'),reviewer_note=? WHERE id=? AND status='pending_review'")
    .bind(reviewerNote,id).run();
  if(Number(update.meta?.changes||0)!==1)
    return json({error:"approval_state_changed",detail:"문서 상태가 변경되었습니다. 목록을 새로고침하세요."},409);
  return json({ok:true,id,status:"approved",reviewer:{provider:identity.provider,email:identity.email,identity_verified:true},license:page.license_name,revision_id:page.revision_id,notice:"Cloudflare Access 인증 검토자와 독립 출처가 기록되었습니다. 사실의 진실성은 별도 검토 대상입니다."});
}
function authorized(request,env){return !!env.ADMIN_TOKEN&&request.headers.get("authorization")==="Bearer "+env.ADMIN_TOKEN;}
export default {
  async fetch(request,env){
    const url=new URL(request.url);
    if(request.method==="GET"&&url.pathname==="/review")return new Response(REVIEW_HTML,{headers:{"content-type":"text/html; charset=utf-8","cache-control":"no-store","content-security-policy":"default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'","referrer-policy":"no-referrer","x-content-type-options":"nosniff"}});
    if(request.method==="GET"&&url.pathname==="/health")return json({ok:true,service:"quantum-wikipedia-explorer",isolated:true,provider:"ko.wikipedia.org"});
    if(!authorized(request,env))return json({error:"unauthorized"},401);
    if(request.method==="GET"&&url.pathname==="/training-status"){
      if(!env.QUANTUM_API_TOKEN||!isCandidateInferenceTarget(env.QUANTUM_INFERENCE_URL))return json({error:"candidate_inference_target_required"},503);
      try{
        const response=await fetch(new URL("/admin/training/status",env.QUANTUM_INFERENCE_URL),{
          headers:{"authorization":"Bearer "+env.QUANTUM_API_TOKEN},signal:AbortSignal.timeout(15000)
        });
        const result=await response.json().catch(()=>({}));
        return json(result,response.status);
      }catch(e){return json({error:"training_service_unreachable",detail:String(e.message||e)},502);}
    }
    if(request.method==="GET"&&url.pathname==="/page"){
      const id=String(url.searchParams.get("id")||"").trim();
      if(!id)return json({error:"id_required"},400);
      const page=await env.DB.prepare("SELECT id,provider,title,url,domain,content,snippet,status,license_status,license_name,attribution,revision_id,content_hash,reviewer_note,discovered_at FROM pages WHERE id=?").bind(id).first();
      if(!page)return json({error:"page_not_found"},404);
      return json({page_id:page.id,...page,content:page.content||""});
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
    if(request.method==="GET"&&url.pathname==="/export-candidate"){
      const corpus=String(url.searchParams.get("corpus")||"unverified");
      if(!["manual","unverified"].includes(corpus))return json({error:"corpus_must_be_manual_or_unverified"},400);
      const limit=Math.min(10,Math.max(1,Number(url.searchParams.get("limit")||5)));
      const after=String(url.searchParams.get("after")||"");
      const eligibility=corpus==="manual"
        ? "status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%'"
        : "status='pending_review'";
      const rows=await env.DB.prepare("SELECT id,title,url,content,content_hash,revision_id,license_name,attribution,status,reviewer_note,strftime('%Y-%m-%dT%H:%M:%fZ',reviewed_at) AS reviewed_at FROM pages WHERE "+eligibility+" AND id>? ORDER BY id LIMIT ?").bind(after,limit).all();
      const documents=(rows.results||[]).map(row=>({
        page_id:row.id,title:row.title,url:row.url,text:String(row.content||""),
        license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors",
        revision_id:row.revision_id,content_hash:row.content_hash,status:row.status,
        reviewer_note:row.reviewer_note||"",reviewed_at:row.reviewed_at||null,
        provenance_review:parseVerifiedReviewNote(row.reviewer_note,row.reviewed_at)
      })).filter(row=>row.text.length>=200);
      return json({corpus,documents,next_after:rows.results?.length?rows.results[rows.results.length-1].id:null,has_more:(rows.results||[]).length===limit});
    }
    if(request.method==="GET"&&url.pathname==="/export-approved"){
      // Legacy export route is deliberately restricted to explicit human approvals.
      const limit=Math.min(10,Math.max(1,Number(url.searchParams.get("limit")||5)));
      const after=String(url.searchParams.get("after")||"");
      const rows=await env.DB.prepare("SELECT id,title,url,content,content_hash,revision_id,license_name,attribution,reviewer_note,strftime('%Y-%m-%dT%H:%M:%fZ',reviewed_at) AS reviewed_at FROM pages WHERE status='approved' AND reviewer_note LIKE 'MANUAL_APPROVAL_V2:%' AND id>? ORDER BY id LIMIT ?").bind(after,limit).all();
      const documents=(rows.results||[]).map(row=>({
        page_id:row.id,title:row.title,url:row.url,text:String(row.content||""),
        license:row.license_name||"CC BY-SA 4.0",attribution:row.attribution||"Wikipedia contributors",
        revision_id:row.revision_id,content_hash:row.content_hash,status:"approved",
        reviewer_note:row.reviewer_note||"",
        reviewed_at:row.reviewed_at||null,
        provenance_review:parseVerifiedReviewNote(row.reviewer_note,row.reviewed_at)
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
    if(request.method==="POST"&&url.pathname==="/approve"){
      const verification=await verifyCloudflareAccessIdentity(request,env);
      if(!verification.ok)return json({error:verification.error},403);
      return approvePage(request,env,verification.identity);
    }
    if(request.method==="GET"&&url.pathname==="/sync-preview")return json(await previewApprovedSync(env));
    if(request.method==="POST"&&url.pathname==="/sync-approved")return json(await syncApproved(env));
    return json({error:"not_found"},404);
  },
  async scheduled(_event,env,ctx){ctx.waitUntil(runCycle(env));}
};
