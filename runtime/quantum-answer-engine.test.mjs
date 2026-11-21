import test from "node:test";
import assert from "node:assert/strict";
import { createQuantumAnswerEngine } from "./quantum-answer-engine.mjs";

test("passes relevant source evidence to Quantum and returns server-authored citations", async () => {
  let generationRequest;
  const engine = createQuantumAnswerEngine({
    retrieveDocuments: async query => [
      {
        title: "Cloudflare Workers 문서",
        url: "https://developers.cloudflare.com/workers/",
        publisher: "Cloudflare",
        license: "Cloudflare documentation terms",
        attribution: "Cloudflare",
        text: "Cloudflare Workers는 JavaScript 기반 서버리스 실행 환경입니다.",
      },
      {
        title: "무관한 요리 문서",
        url: "https://example.org/cooking",
        text: "김치찌개 조리법과 재료에 대한 내용입니다.",
      },
      {
        title: "잘못된 URL",
        url: "javascript:alert(1)",
        text: "Cloudflare Workers 정보",
      },
    ],
    generateWithQuantum: async request => {
      generationRequest = request;
      return { text: "Cloudflare Workers는 서버리스 실행 환경입니다. [S1]", model: "Quantum-Candidate" };
    },
  });

  const result = await engine({
    question: "Cloudflare Workers란 무엇인가요?",
    history: [{ role: "user", content: "서버에 대해 알려줘" }],
  });

  assert.match(generationRequest.messages[0].content, /You are Quantum/);
  assert.match(generationRequest.messages[0].content, /\[SOURCE S1\]/);
  assert.equal(result.model, "Quantum-Candidate");
  assert.equal(result.source, "juckex-quantum");
  assert.equal(result.sources.length, 1);
  assert.equal(result.sources[0].url, "https://developers.cloudflare.com/workers/");
  assert.equal(result.retrieval.suppliedToModel, 1);
});

test("does not invent source citations when retrieval has no relevant documents", async () => {
  let systemPrompt = "";
  const engine = createQuantumAnswerEngine({
    retrieveDocuments: async () => [{
      title: "Unrelated",
      url: "https://example.org/unrelated",
      text: "completely different content about gardening and flowers",
    }],
    generateWithQuantum: async ({ messages }) => {
      systemPrompt = messages[0].content;
      return "현재 제공된 자료만으로는 확인할 수 없습니다.";
    },
  });
  const result = await engine({ question: "양자컴퓨터의 오류 정정 방식" });
  assert.equal(result.sources.length, 0);
  assert.equal(result.retrieval.suppliedToModel, 0);
  assert.match(systemPrompt, /No sufficiently relevant external documents/);
});

test("rejects missing question and missing Quantum generator", async () => {
  assert.throws(() => createQuantumAnswerEngine({ retrieveDocuments: async () => [] }), /generateWithQuantum/);
  const engine = createQuantumAnswerEngine({
    retrieveDocuments: async () => [],
    generateWithQuantum: async () => "ok",
  });
  await assert.rejects(() => engine({ question: " " }), /question is required/);
});

test("does not accept source URLs using non-web schemes", async () => {
  const engine = createQuantumAnswerEngine({
    retrieveDocuments: async () => [{
      title: "Bad source",
      url: "file:///etc/passwd",
      text: "Quantum model documentation",
    }],
    generateWithQuantum: async () => "답변",
  });
  const result = await engine({ question: "Quantum model documentation" });
  assert.equal(result.sources.length, 0);
});
