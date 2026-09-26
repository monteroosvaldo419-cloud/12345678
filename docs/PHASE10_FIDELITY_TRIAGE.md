# Phase 10 Fidelity Triage

The previous broad run reported 21 failures. They do not all indicate an engine
bug. The table records the classification before changing mechanics.

Legend: A = simulator defect, B = data/loader gap or version mismatch, C =
expectation drift, D = deliberate documented approximation, E = infrastructure.

| Test | Category | Cause | File | Action |
|---|---|---|---|---|
| `test_log_changes_only_the_group_inside_its_area` | C | Initial test asserted immediate damage although Log is rolling/delayed | `tests/test_sim_causal_scenarios.py` | Fixed test to advance impact |
| `test_wait_advances_time_and_elixir_without_changing_cycle` | C | Fixture started at elixir cap | `tests/test_sim_causal_scenarios.py` | Fixed fixture to start at 5 elixir |
| `test_a_lone_ice_golem_reaches_the_tower_and_lands_one_hit` | B | Current extracted Ice Golem/tower values differ from the old live anchor | `sim/gamedata.py`, `tests/test_sim_fidelity.py` | Keep as calibration gap; no invented values |
| `test_a_dying_unit_explodes_on_enemies_and_spares_its_own_side` | B | Current `ice_golemite` data has no death blast fields | `sim/gamedata.py` | Mark unknown until source data is found |
| `test_spell_combat_numbers_scale_with_the_requested_card_level` | B | Current Zap spell source does not populate damage | `sim/spells.py` | Mark missing source data |
| `test_earthquake_hits_ground_and_deals_its_building_multiplier` | B | Current spell flags/data disagree with the historical ground-only expectation | `sim/spells.py` | Audit source before changing behavior |
| `test_vines_selects_three_targets_and_applies_its_snare_area` | B | Extracted duration is 1400 ms, test expects older 2000 ms | `sim/spells.py` | Preserve extracted value; update evidence later |
| `test_clone_makes_one_hitpoint_friendly_troops_only_once` | B | `OnlyOwnTroops` is absent/false in current spell extraction | `sim/spells.py` | Mark loader/data gap |
| `test_electro_wizard_deployment_aeo_uses_its_source_action_graph` | B | Deployment AEO fields are not populated from current character data | `sim/gamedata.py` | Mark extraction gap |
| `test_evolved_bat_on_hit_heal_scales_and_can_overheal_to_200_percent` | B | Current evolution overlay values differ, including negative heal | `sim/gamedata.py` | Do not override without versioned evidence |
| `test_evolved_skeleton_duplicates_on_hit_but_respects_group_cap` | B | Current overlay lacks the spawn action fields | `sim/gamedata.py` | Mark extraction gap |
| `test_evolved_recruit_unlocks_client_charge_when_shield_breaks` | B | Shield-break charge range is absent in current data | `sim/gamedata.py` | Mark unknown |
| `test_evolved_wizard_shield_break_explodes_with_current_damage_and_push` | B | Current level/evolution shield value differs from historical assertion | `sim/gamedata.py` | Version the expected value before changing |
| `test_evolved_archer_uses_current_power_shot_damage_beyond_four_point_five_tiles` | B | Far-attack minimum range is absent | `sim/gamedata.py` | Mark extraction gap |
| `test_evolved_ice_spirit_repeats_target_bound_area_freeze_after_three_seconds` | B | Target-area delay is absent in current extracted data | `sim/gamedata.py` | Mark extraction gap |
| `test_evolved_royal_hog_descends_then_deals_current_ground_only_landing_damage` | B | Ground transition HP threshold is absent | `sim/gamedata.py` | Mark extraction gap |
| `test_hero_berserker_savage_survival_cast_and_current_bear_buff` | B | Current ability duration is 3500 ms, assertion expects 4000 ms | `sim/gamedata.py` | Preserve current source value |
| `test_hero_dark_prince_dismount_preserves_damage_and_spawns_current_rhino` | A | Form transition drops shield state | `sim/engine.py` | Isolated engine defect; defer to fidelity fix |
| `test_hero_wizard_fiery_flight_changes_form_and_adds_enhanced_hit_area` | B | Current hero form damage differs from historical value | `sim/gamedata.py` | Version/data audit required |
| `test_coverage_has_no_unresolved_public_spell_and_counts_evo_summons_as_units` | C | Coverage expectation is 176 while current loader finds 178 | `tests/test_sim_fidelity.py` | Update expectation after data audit |
| `test_the_mask_does_not_hide_a_legal_action` | E | Deepcopy attempted to copy open shadow JSONL stream | `scripts/brain/policy.py` | Fixed with copy-safe runtime handle isolation |

No new combat constants were invented. B-category items remain explicit data
or extraction gaps rather than being silently patched with remembered values.
