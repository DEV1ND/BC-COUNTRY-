import test from "node:test";
import assert from "node:assert/strict";
import { planMissingPageIds } from "../knowledge_sync_plan.mjs";
test("selects only absent IDs",()=>assert.deepEqual(planMissingPageIds(["a","b","c"],new Set(["a"]),5),["b","c"]));
test("respects stable batch limit",()=>assert.deepEqual(planMissingPageIds(["z","y","x"],new Set(),2),["z","y"]));
test("deduplicates IDs",()=>assert.deepEqual(planMissingPageIds(["a","a","b"],new Set(),5),["a","b"]));
test("returns empty when cache is complete",()=>assert.deepEqual(planMissingPageIds(["a","b"],new Set(["a","b"]),5),[]));
