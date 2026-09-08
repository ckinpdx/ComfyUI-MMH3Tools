"""RefMod insertion and compatibility checks without a model or mod files."""

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[3])]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
from mmh3tools import NODES
from mmh3tools.nodes_refmods import H3RefModCondSetApply as Apply

torch.set_num_threads(2)


class Mod:
    name = "example"
    mode = "encode"
    token_count = 4

    def __init__(self):
        self.strengths = []

    def ref_block(self, strength):
        self.strengths.append(strength)
        if strength <= 0:
            return None
        return {"kind": "video", "latent": torch.full((1, 24, 5, 4, 4), strength)}


def cond_set():
    tokens = torch.ones(1, 4, 16)
    refs = [{"kind": kind} for kind in ("image", "image", "audio", "video_audio")]
    metadata = {"minimax_refs": refs, "pooled_output": tokens, "marker": "unchanged"}
    return {"conds": [[[tokens, metadata]], [[tokens, metadata], [tokens, {}]]],
            "prompts": ["first", "second"], "fingerprint": "original"}


def apply(source, mods=None, retention=1., position="before_controls", override=-1):
    return Apply.execute(source, retention, position, override, mods)


class RefModTests(unittest.TestCase):
    def test_saved_workflow_schema_and_registration(self):
        schema = Apply.GET_SCHEMA()
        self.assertEqual(schema.node_id, "H3RefModCondSetApply")
        self.assertEqual(schema.category, "MMH3Tools/conditioning")
        inputs = Apply.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]),
                         ["cond_set", "retention", "insert_position", "controls_override"])
        self.assertEqual(list(inputs["optional"]), ["mods"])
        self.assertEqual(inputs["required"]["cond_set"][0], "MMH3_COND_SET")
        self.assertEqual(inputs["optional"]["mods"][0], "H3_REF_MODS")
        self.assertEqual(inputs["required"]["retention"][1]["default"], 1.)
        self.assertEqual(inputs["required"]["insert_position"][1]["options"],
                         ["before_controls", "append_last"])
        self.assertEqual(inputs["required"]["controls_override"][1]["default"], -1)
        self.assertEqual(list(Apply.RETURN_TYPES), ["MMH3_COND_SET", "STRING"])
        self.assertEqual(list(Apply.RETURN_NAMES), ["cond_set", "report"])
        self.assertEqual([n for n in NODES if n is Apply], [Apply])

    def test_inactive_bundle_is_exact_passthrough(self):
        source, mod = cond_set(), Mod()
        for mods, retention in [(None, 1.), ([], 1.), ([(mod, 1.)], 0.), ([(mod, 0.)], 1.)]:
            result = apply(source, mods, retention)
            self.assertIs(result[0], source)
            self.assertIn("passthrough", result[1])
        self.assertIsNone(apply(None)[0])

    def test_all_entries_receive_mods_without_mutating_inputs(self):
        source, mod = cond_set(), Mod()
        result = apply(source, [(mod, 1.)])[0]
        self.assertIs(result["prompts"], source["prompts"])
        self.assertEqual(result["fingerprint"], source["fingerprint"])
        for original_cond, new_cond in zip(source["conds"], result["conds"]):
            for original, new in zip(original_cond, new_cond):
                self.assertIs(new[0], original[0])
                self.assertIsNot(new[1], original[1])
                refs = new[1]["minimax_refs"]
                if "minimax_refs" in original[1]:
                    self.assertEqual([b["kind"] for b in refs],
                                     ["image", "image", "video", "audio", "video_audio"])
                    old_refs = original[1]["minimax_refs"]
                    self.assertEqual(len(old_refs), 4)
                    for old, new_ref in zip(old_refs, refs[:2] + refs[-2:]):
                        self.assertIs(old, new_ref)
                    self.assertIs(new[1]["pooled_output"], original[1]["pooled_output"])
                    self.assertEqual(new[1]["marker"], "unchanged")
                else:
                    self.assertEqual(len(refs), 1)
                    self.assertEqual(original[1], {})
        self.assertIs(result["conds"][0][0][1]["minimax_refs"][2],
                      result["conds"][1][1][1]["minimax_refs"][0])

    def test_strengths_delegate_once_per_loader_row(self):
        mod = Mod()
        result = apply(cond_set(), [(mod, 1.), (mod, .5), (mod, 0.), (mod, 4.)], .5)[0]
        self.assertEqual(mod.strengths, [.5, .25, 0., 1.])
        refs = result["conds"][0][0][1]["minimax_refs"]
        self.assertEqual(len(refs), 7)
        self.assertEqual([b["latent"].flatten()[0].item() for b in refs[2:5]], [.5, .25, 1.])

    def test_append_and_explicit_control_counts(self):
        source, mod = cond_set(), Mod()
        original_refs = source["conds"][0][0][1]["minimax_refs"]
        for position, override, index in [("append_last", -1, 4),
                                          ("before_controls", 0, 4),
                                          ("before_controls", 1, 3),
                                          ("before_controls", 32, 0)]:
            with self.subTest(position=position, override=override):
                result = apply(source, [(mod, 1.)], position=position, override=override)[0]
                refs = result["conds"][0][0][1]["minimax_refs"]
                self.assertIn("latent", refs[index])
                self.assertEqual(refs[:index] + refs[index + 1:], original_refs)

    def test_invalid_bundle_has_actionable_error(self):
        for bundle in [["not-a-loader-row"], [(object(), 1.)]]:
            with self.assertRaisesRegex(ValueError, "loader bundle|ref_block"):
                apply(cond_set(), bundle)

    def test_original_direct_apply_api_is_preserved(self):
        source = cond_set()
        result = Apply().apply(source, 1., "before_controls", -1)
        self.assertIsInstance(result, tuple)
        self.assertIs(result[0], source)
        self.assertEqual(result[1], apply(source)[1])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
