# Stage 0 baseline: `pytest tests/`

- Commit: `de89229` (working tree also had untracked `ridge_field_prototype.py`, not collected by pytest)
- Recorded: 2026-11-09
- Interpreter: `.venv\Scripts\python.exe` (Python 3.12.3) on Windows 11
- Command: `.venv\Scripts\python.exe -m pytest tests/ -rA -q -p no:cacheprovider`
- Result: **13 failed, 117 passed** (130 total). Identical result on two consecutive runs.
- Nothing was fixed. This file is a record only.

## Dependency versions used

- `Flask==3.1.2`
- `numpy==1.26.4`
- `pillow==12.3.0`
- `pypdfium2==5.12.1`
- `pytest==9.1.1`
- `scikit-image==0.26.0`
- `scipy==1.17.1`
- `svgwrite==1.4.3`
- `Werkzeug==3.1.8`

Note: `runtime.txt` specifies python-3.11.13, but the local venv is 3.12. pytest is not in `requirements.txt`.

## Failures (summary)

| File | Failures | Apparent cause (observation only) |
|---|---|---|
| tests/test_recenter_to_crest.py | 6 | **Not tracked by git** (matched by the `test*.py` rule in `.gitignore`, never force-added); it exists only in this local checkout. It references `_recenter_vectorized`, `RECENTER_TO_RIDGE` and `create_fast_paths(gray=...)`, none of which exist in `centerline_engine_light.py`. Likely written against unmerged prototype code. A clean clone would show 7 failed / 117 passed. |
| tests/test_path_optimization.py | 3 | `prune_extracted_paths` assertions on point counts/coordinates no longer match current output. |
| tests/test_web_app_optimization_state.py | 4 | SVG download/generate tests receive `[]` paths or a different curve-fitting default than expected. |

## Per-test results

| File | Test | Result | Message |
|---|---|---|---|
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_horizontal_line[300]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_horizontal_line[800]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_horizontal_line[1500]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_branching_field[300]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_branching_field[800]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_fixture_skeletons[corner-l-shape]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_fixture_skeletons[curve-true-circle]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_matches_legacy_fixture_skeletons[square-perfect]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_extract_skeleton_paths_uses_vectorized_walker_above_threshold` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_extract_skeleton_paths_uses_legacy_walker_below_threshold` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_extract_skeleton_paths_flag_toggle_preserves_fixture_summary[corner-l-shape]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_extract_skeleton_paths_flag_toggle_preserves_fixture_summary[curve-true-circle]` | passed |  |
| tests/test_fast_paths_vectorized.py | `test_create_fast_paths_vectorized_benchmark_large_branching_field` | passed |  |
| tests/test_line_only_fit.py | `test_line_only_fit_follows_small_closed_loops` | passed |  |
| tests/test_path_optimization.py | `test_precondition_path_for_optimization_simplifies_staircase_noise` | passed |  |
| tests/test_path_optimization.py | `test_precondition_path_for_optimization_projects_through_terminal_spurs` | passed |  |
| tests/test_path_optimization.py | `test_precondition_path_for_optimization_collapses_monotone_staircase_line` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_removes_terminal_spur_artifacts` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_collapses_staircase_noise_before_fitting` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_drops_projected_terminal_spur_endpoints` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_drops_longer_shallow_terminal_runs` | passed |  |
| tests/test_path_optimization.py | `test_critical_path_indices_ignore_line_like_micro_inflections` | passed |  |
| tests/test_path_optimization.py | `test_critical_path_indices_keep_stronger_sawtooth_inflections` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_keeps_multiple_points_for_curves` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_can_return_debug_diagnostics` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_reports_unchanged_path_reasons` | passed |  |
| tests/test_path_optimization.py | `test_build_line_span_hints_marks_two_point_path_as_full_line` | passed |  |
| tests/test_path_optimization.py | `test_build_line_span_hints_marks_collinear_anchor_run` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_preserves_closed_loop_tangent_anchors_for_true_circle` | FAILED | assert 87 <= 45 |
| tests/test_path_optimization.py | `test_prune_extracted_paths_removes_tiny_square_corner_arising_segments` | passed |  |
| tests/test_path_optimization.py | `test_prune_extracted_paths_removes_tiny_open_corner_arising_segment` | FAILED | assert [(54.0, 53.0)...3333334), ...] == [(54.0, 52.0)...), (9.0, 9.0)] |
| tests/test_path_optimization.py | `test_prune_extracted_paths_removes_rotated_open_corner_cluster` | FAILED | assert 17 == 3 |
| tests/test_path_optimization.py | `test_optimizer_allows_two_point_result_for_long_open_staircase` | passed |  |
| tests/test_path_optimization.py | `test_optimizer_rejects_self_intersecting_candidate` | passed |  |
| tests/test_path_optimization.py | `test_optimizer_skips_arc_candidates_for_straight_paths` | passed |  |
| tests/test_path_optimization.py | `test_merge_nearby_paths_rejects_self_intersecting_bridge` | passed |  |
| tests/test_path_optimization.py | `test_merge_nearby_paths_reports_metrics_and_progress` | passed |  |
| tests/test_path_optimization.py | `test_merge_nearby_paths_skips_any_long_candidate_in_cheap_pass` | passed |  |
| tests/test_path_optimization.py | `test_merge_nearby_paths_skips_long_long_candidates_in_cheap_pass` | passed |  |
| tests/test_path_optimization.py | `test_merge_nearby_paths_long_only_pass_can_merge_long_fragments` | passed |  |
| tests/test_preprocessing_profiles.py | `test_faint_line_art_gets_profile_gated_enhancement` | passed |  |
| tests/test_preprocessing_profiles.py | `test_dark_line_art_stays_on_standard_profile` | passed |  |
| tests/test_preprocessing_profiles.py | `test_ultra_faint_uniform_line_art_forces_faint_profile` | passed |  |
| tests/test_recenter_to_crest.py | `test_recenter_vectorized_moves_open_path_toward_dark_crest_and_preserves_endpoints` | FAILED | AttributeError: module 'centerline_engine_light' has no attribute '_recenter_vectorized' |
| tests/test_recenter_to_crest.py | `test_recenter_vectorized_keeps_parallel_stroke_midline_stable_when_dark_choices_are_bilateral` | FAILED | AttributeError: module 'centerline_engine_light' has no attribute '_recenter_vectorized' |
| tests/test_recenter_to_crest.py | `test_recenter_vectorized_freezes_points_inside_no_recenter_zone_but_keeps_clean_runs_mobile` | FAILED | AttributeError: module 'centerline_engine_light' has no attribute '_recenter_vectorized' |
| tests/test_recenter_to_crest.py | `test_fast_path_recentering_preserves_path_summary_for_legacy_and_vectorized_walkers` | FAILED | TypeError: create_fast_paths() got an unexpected keyword argument 'gray' |
| tests/test_recenter_to_crest.py | `test_extract_skeleton_paths_recenter_flag_is_opt_in_for_raw_callers` | FAILED | AttributeError: module 'centerline_engine_light' has no attribute 'RECENTER_TO_RIDGE' |
| tests/test_recenter_to_crest.py | `test_recenter_timing_prints_mid_band_and_large_case_metrics` | FAILED | AttributeError: module 'centerline_engine_light' has no attribute '_recenter_vectorized' |
| tests/test_resolution_scaling.py | `test_resolution_scale_is_neutral_for_typical_images` | passed |  |
| tests/test_resolution_scaling.py | `test_effective_parameters_expand_for_large_images` | passed |  |
| tests/test_resolution_scaling.py | `test_stroke_width_can_reduce_merge_gap_for_large_thin_lines` | passed |  |
| tests/test_resolution_scaling.py | `test_stroke_factor_mode_uses_merge_gap_as_stroke_multiplier` | passed |  |
| tests/test_resolution_scaling.py | `test_mask_based_stroke_width_estimate_uses_skeleton_distance` | passed |  |
| tests/test_resolution_scaling.py | `test_profile_cap_applies_before_resolution_scaling` | passed |  |
| tests/test_resolution_scaling.py | `test_effective_min_object_size_is_capped_by_logical_min_path_length` | passed |  |
| tests/test_resolution_scaling.py | `test_effective_min_object_size_relaxes_for_large_images` | passed |  |
| tests/test_resolution_scaling.py | `test_auto_tune_returns_logical_min_length_with_scaled_effective_length` | passed |  |
| tests/test_resolution_scaling.py | `test_random_tile_mosaic_uses_larger_tiles_for_large_images` | passed |  |
| tests/test_resolution_scaling.py | `test_sampling_plan_increases_tile_budget_for_large_images` | passed |  |
| tests/test_resolution_scaling.py | `test_random_tile_mosaic_places_hotspot_tiles_over_active_regions` | passed |  |
| tests/test_resolution_scaling.py | `test_random_tile_mosaic_scatter_coverage_tiles_away_from_perimeter_only` | passed |  |
| tests/test_resolution_scaling.py | `test_auto_detect_dark_threshold_uses_preview_sampling_for_large_images` | passed |  |
| tests/test_resolution_scaling.py | `test_auto_detect_dark_threshold_respects_cancellation_before_work` | passed |  |
| tests/test_small_image_supersampling.py | `test_downscale_extracted_paths_round_trips_known_supersampled_path` | passed |  |
| tests/test_small_image_supersampling.py | `test_downscale_extracted_paths_is_noop_at_scale_one` | passed |  |
| tests/test_small_image_supersampling.py | `test_resolve_small_image_supersample_factor_extends_into_mid_band_but_skips_large_images` | passed |  |
| tests/test_small_image_supersampling.py | `test_extract_skeleton_paths_downscales_legacy_results_when_supersampling_is_active` | passed |  |
| tests/test_small_image_supersampling.py | `test_extract_skeleton_paths_scales_min_path_length_with_supersample_factor` | passed |  |
| tests/test_small_image_supersampling.py | `test_extract_skeleton_paths_scales_min_object_size_with_supersample_factor` | passed |  |
| tests/test_small_image_supersampling.py | `test_extract_skeleton_paths_keeps_native_thresholds_for_large_images` | passed |  |
| tests/test_small_image_supersampling.py | `test_supersampled_speckle_filter_is_not_more_permissive_than_native_run` | passed |  |
| tests/test_small_image_supersampling.py | `test_supersample_timing_prints_mid_band_and_large_case_metrics` | passed |  |
| tests/test_small_image_supersampling.py | `test_small_object_filter_keeps_one_pixel_diagonal_lines_but_drops_corner_specks` | passed |  |
| tests/test_web_app_optimization_state.py | `test_benchmark_filename_prefix_detection` | passed |  |
| tests/test_web_app_optimization_state.py | `test_debug_filename_marker_detection` | passed |  |
| tests/test_web_app_optimization_state.py | `test_process_centerlines_force_enables_pruning_debug_for_debug_filename` | passed |  |
| tests/test_web_app_optimization_state.py | `test_execute_test_run_passes_fitting_parameters` | passed |  |
| tests/test_web_app_optimization_state.py | `test_test_ui_backend_normalizes_main_app_analysis_parameters` | passed |  |
| tests/test_web_app_optimization_state.py | `test_execute_test_run_rejects_non_object_fitting_parameters` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_tune_tile_preview_reuses_cached_tile_entries` | passed |  |
| tests/test_web_app_optimization_state.py | `test_benchmark_metrics_route_returns_filename_triggered_metrics` | passed |  |
| tests/test_web_app_optimization_state.py | `test_benchmark_metrics_route_rejects_non_benchmark_session` | passed |  |
| tests/test_web_app_optimization_state.py | `test_upload_response_exposes_benchmark_metrics_url_for_triggered_filename` | passed |  |
| tests/test_web_app_optimization_state.py | `test_upload_remote_fetches_http_image` | passed |  |
| tests/test_web_app_optimization_state.py | `test_extract_tile_preview_paths_returns_all_kept_paths_without_rank` | passed |  |
| tests/test_web_app_optimization_state.py | `test_upload_remote_falls_back_from_html_page_to_og_image` | passed |  |
| tests/test_web_app_optimization_state.py | `test_upload_remote_rejects_non_http_scheme` | passed |  |
| tests/test_web_app_optimization_state.py | `test_progress_reports_failed_optimization` | passed |  |
| tests/test_web_app_optimization_state.py | `test_progress_reports_live_preview_metadata` | passed |  |
| tests/test_web_app_optimization_state.py | `test_progress_can_skip_live_preview_path_payload` | passed |  |
| tests/test_web_app_optimization_state.py | `test_centerline_session_defaults_to_disabled_optimization` | passed |  |
| tests/test_web_app_optimization_state.py | `test_process_immediate_reports_disabled_optimization_state` | passed |  |
| tests/test_web_app_optimization_state.py | `test_process_immediate_skips_path_simplification_when_disabled` | passed |  |
| tests/test_web_app_optimization_state.py | `test_process_centerlines_skips_pruning_when_path_simplification_disabled` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_detect_threshold_success_keeps_optimization_disabled` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_tune_extraction_success_keeps_optimization_disabled` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_immediate_disabled_optimization_uses_direct_single_layer` | passed |  |
| tests/test_web_app_optimization_state.py | `test_download_svg_disabled_optimization_uses_direct_single_layer` | FAILED | assert [] == [[[1, 1], [1, 2], [1, 3]]] |
| tests/test_web_app_optimization_state.py | `test_download_svg_final_results_disabled_optimization_uses_direct_single_layer` | FAILED | assert [] == [[[1, 1], [1, 2], [1, 3]]] |
| tests/test_web_app_optimization_state.py | `test_generate_svg_disabled_optimization_can_enable_post_fit_export` | passed |  |
| tests/test_web_app_optimization_state.py | `test_progress_reports_merge_progress_metadata` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_tune_progress_allows_running_state_past_90_seconds` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_tune_progress_reports_terminal_timeout_from_server_state` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_tune_sets_merge_gap_to_two_times_stroke_factor` | passed |  |
| tests/test_web_app_optimization_state.py | `test_auto_detect_threshold_start_runs_background_job_and_updates_session` | passed |  |
| tests/test_web_app_optimization_state.py | `test_stop_auto_detect_marks_progress_cancelled_without_overwriting_threshold` | passed |  |
| tests/test_web_app_optimization_state.py | `test_background_optimization_records_optimizer_phase_metrics` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_final_works_immediately_after_background_optimization` | passed |  |
| tests/test_web_app_optimization_state.py | `test_benchmark_response_helper_returns_none_when_disabled` | passed |  |
| tests/test_web_app_optimization_state.py | `test_benchmark_response_helper_includes_compact_summary` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_live_preview_includes_blue_when_under_limit` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_progressive_uses_live_merge_preview_before_optimization` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_progressive_stitches_optimized_prefix_with_live_preview_tail` | passed |  |
| tests/test_web_app_optimization_state.py | `test_download_svg_includes_completed_blue_and_selected_magenta_and_image` | passed |  |
| tests/test_web_app_optimization_state.py | `test_download_svg_query_overrides_include_image` | FAILED | assert [] == [[[0, 0], [0, 1], [0, 2]]] |
| tests/test_web_app_optimization_state.py | `test_download_svg_prefers_final_results_over_empty_progressive_state_when_complete` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_samples_preview_when_detected_paths_exceed_limit` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_reuses_cached_output_for_identical_request` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_cache_invalidates_when_geometry_changes_with_same_counts` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_defaults_curve_fitting_off` | FAILED | assert True is False |
| tests/test_web_app_optimization_state.py | `test_generate_svg_preview_cache_invalidates_when_curve_fitting_toggles` | passed |  |
| tests/test_web_app_optimization_state.py | `test_generate_svg_view_and_download_share_render_settings` | passed |  |
| tests/test_web_app_optimization_state.py | `test_release_session_frees_heavy_data` | passed |  |
| tests/test_web_app_optimization_state.py | `test_session_lookup_refreshes_last_accessed` | passed |  |
| tests/test_web_app_optimization_state.py | `test_evict_stale_sessions_removes_idle_sessions` | passed |  |
| tests/test_web_app_optimization_state.py | `test_evict_stale_sessions_protects_active_work_under_cap` | passed |  |
| tests/test_web_app_optimization_state.py | `test_evict_enforces_max_sessions_cap` | passed |  |
| tests/test_web_app_optimization_state.py | `test_upload_releases_replaced_session` | passed |  |
| tests/test_web_app_optimization_state.py | `test_bounded_progress_queue_drops_oldest_when_full` | passed |  |
