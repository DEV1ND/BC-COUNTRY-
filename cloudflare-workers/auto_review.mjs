// Conservative screening for the isolated Quantum Wikipedia crawler.
// Automated checks are NOT factual verification and can never approve a page.
const ADULT = /포르노|성인.?영상|성행위|성적.?착취|아동.?성착취/i;
const LOW_VALUE = /고등학교|중학교|초등학교|기업|주식회사|그룹 AG|영화|소설|음반|노래|선거구|역대|목록|분류|연도별|동음이의어/i;
const SENSITIVE = /주민등록번호|여권번호|전화번호|휴대전화|이메일 주소|집 주소|개인정보/i;
const TOPIC = /(인공지능|학|과학|수학|물리|화학|생물|천문|지질|경제|정치|사회|언어|철학|역사|기술|공학|컴퓨터|정보|알고리즘|자료구조|환경|기후|교육|법학|논리|문법|통계|확률|금융|건축|에너지|생태|의학|해부|생리|심리|문학|지리|우주|원자|분자|유전|진화|민주주의|국제 관계)/;

export function reviewCandidate(row) {
  const title = String(row.title || "").trim();
  const text = String(row.content || "").trim();
  const url = String(row.url || "");
  const domain = String(row.domain || "");
  if (ADULT.test(title + " " + text))
    return { status: "excluded", score: 0, reason: "성인 콘텐츠 주제 제외" };
  if (SENSITIVE.test(text))
    return { status: "pending_review", score: 0, reason: "개인정보 가능성으로 사람 검토 필요" };
  if (domain !== "ko.wikipedia.org" || !url.startsWith("https://ko.wikipedia.org/wiki/"))
    return { status: "pending_review", score: 0, reason: "허용된 출처 도메인 확인 실패" };
  if (row.license_status !== "attribution_required" || row.license_name !== "CC BY-SA 4.0" ||
      !String(row.attribution || "").trim() || !row.revision_id ||
      !/^[a-f0-9]{64}$/i.test(String(row.content_hash || "")))
    return { status: "pending_review", score: 0, reason: "라이선스·출처·리비전·해시 메타데이터 불충분" };

  let score = 30;
  if (text.length >= 1800 && text.length <= 50000) score += 20;
  else return { status: "pending_review", score, reason: "본문 길이 또는 완결성 검토 필요" };
  if (LOW_VALUE.test(title))
    return { status: "pending_review", score, reason: "개별 인물·기관·작품·목록형 항목은 우선 대기" };
  if (TOPIC.test(title)) score += 20;
  else return { status: "pending_review", score, reason: "범용 지식 주제인지 분류 불확실" };
  if (/(은|는|이란|란|분야이다|학문이다|방법이다|체계이다|개념이다|연구한다|가리킨다|말한다)/.test(text.slice(0, 500))) score += 15;
  else return { status: "pending_review", score, reason: "정의형 도입부 확인 필요" };
  const paragraphs = text.split(/\n{2,}/).filter(x => x.trim().length > 50).length;
  if (paragraphs >= 3) score += 10;
  else return { status: "pending_review", score, reason: "본문 구조가 짧거나 불완전함" };
  const segments = text.split(/[.!?。\n]/).filter(x => x.trim().length > 20);
  const unique = new Set(segments.map(x => x.trim()));
  const duplicateRatio = segments.length ? 1 - unique.size / segments.length : 1;
  if (duplicateRatio > 0.25)
    return { status: "pending_review", score, reason: "중복 문장 비율이 높음" };
  score += 5;
  return {
    status: "pending_review",
    score,
    reason: "자동 형식 검사 통과. 사실 검증과 사람의 승인 전까지 지식 인덱스에 반영하지 않음.",
  };
}

export async function validateManualApproval(row, reviewerNote) {
  const note = String(reviewerNote || "").trim();
  if (note.length < 20)
    return { ok: false, reason: "사실 확인 요약을 20자 이상 작성해야 합니다." };
  const referenceUrls = note.match(/https:\/\/[^\s<>"']+/g) || [];
  const hasIndependentReference = referenceUrls.some(value => {
    try {
      const reference = new URL(value.replace(/[),.;]+$/, ""));
      return reference.protocol === "https:" &&
        reference.hostname !== "ko.wikipedia.org" &&
        !reference.hostname.endsWith(".wikipedia.org");
    } catch {
      return false;
    }
  });
  if (!hasIndependentReference)
    return { ok: false, reason: "위키백과 외부의 독립적인 HTTPS 대조 출처를 검토 메모에 포함해야 합니다." };
  const url = String(row.url || "");
  if (row.provider !== "wikipedia" || row.domain !== "ko.wikipedia.org" ||
      !url.startsWith("https://ko.wikipedia.org/wiki/"))
    return { ok: false, reason: "위키백과 원문 URL과 출처 도메인을 확인할 수 없습니다." };
  if (row.license_status !== "attribution_required" || row.license_name !== "CC BY-SA 4.0" ||
      !String(row.attribution || "").trim())
    return { ok: false, reason: "CC BY-SA 4.0 라이선스와 저작자 표시가 필요합니다." };
  if (!String(row.revision_id || "").trim() ||
      !/^[a-f0-9]{64}$/i.test(String(row.content_hash || "")))
    return { ok: false, reason: "원문 리비전 ID 또는 SHA-256 해시가 없습니다." };
  const body = String(row.content || "").trim();
  if (body.length < 200 || body.length > 50000)
    return { ok: false, reason: "본문 길이는 200~50000자여야 합니다." };
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body));
  const actualHash = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join("");
  if (actualHash !== String(row.content_hash || "").toLowerCase())
    return { ok: false, reason: "저장된 본문과 SHA-256 해시가 일치하지 않습니다." };
  if (SENSITIVE.test(String(row.title || "") + " " + body))
    return { ok: false, reason: "개인정보 가능성이 있어 승인할 수 없습니다." };
  return { ok: true, reason: "수동 검토 메모 및 출처 메타데이터 확인 완료" };
}
