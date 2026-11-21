import test from "node:test";
import assert from "node:assert/strict";
import { planMissingPageIds } from "../knowledge_sync_plan.mjs";

test("selects only absent IDs",()=>assert.deepEqual(planMissingPageIds(["a","b","c"],new Set(["a"]),5),["b","c"]));
test("respects the requested batch limit",()=>assert.deepEqual(planMissingPageIds(["z","y","x"],new Set(),2),["z","y"]));
test("deduplicates IDs",()=>assert.deepEqual(planMissingPageIds(["a","a","b"],new Set(),5),["a","b"]));
test("returns empty when cache is complete",()=>assert.deepEqual(planMissingPageIds(["a","b"],new Set(["a","b"]),5),[]));
test("returns empty for an empty approved manifest",()=>assert.deepEqual(planMissingPageIds([],new Set(),5),[]));
test("zero and negative limits clamp to one",()=>{
  assert.deepEqual(planMissingPageIds(["a","b"],new Set(),0),["a"]);
  assert.deepEqual(planMissingPageIds(["a","b"],new Set(),-3),["a"]);
});
test("non-finite limits use the default batch size",()=>assert.equal(planMissingPageIds(Array.from({length:8},(_,i)=>String(i)),new Set(),Number.NaN).length,5));
test("configured batch size is capped at fifty",()=>assert.equal(planMissingPageIds(Array.from({length:80},(_,i)=>String(i)),new Set(),500).length,50));
test("normalizes numeric IDs to strings",()=>assert.deepEqual(planMissingPageIds([10,20],new Set(["10"]),5),["20"]));
