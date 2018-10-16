#include "../gtest.h"

#include <arbor/common_types.hpp>
#include <arbor/mc_cell.hpp>

#include "backends/event.hpp"
#include "backends/multicore/fvm.hpp"
#include "fvm_lowered_cell_impl.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"
#include <arbor/util/any.hpp>


using namespace arb;
using fvm_cell = fvm_lowered_cell_impl<multicore::backend>;
using shared_state = multicore::backend::shared_state;

ACCESS_BIND(std::unique_ptr<shared_state> fvm_cell::*, fvm_state_ptr, &fvm_cell::state_);

TEST(probe, fvm_lowered_cell) {
    execution_context context;

    mc_cell bs = make_cell_ball_and_stick(false);

    i_clamp stim(0, 100, 0.3);
    bs.add_stimulus({1, 1}, stim);

    cable1d_recipe rec(bs);

    segment_location loc0{0, 0};
    segment_location loc1{1, 1};
    segment_location loc2{1, 0.3};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 20, cell_probe_address{loc1, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 30, cell_probe_address{loc2, cell_probe_address::membrane_current});

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0}, rec, targets, probe_map);

    EXPECT_EQ(3u, rec.num_probes(0));
    EXPECT_EQ(3u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({0, 1}).tag);
    EXPECT_EQ(30, probe_map.at({0, 2}).tag);

    probe_handle p0 = probe_map.at({0, 0}).handle;
    probe_handle p1 = probe_map.at({0, 1}).handle;
    probe_handle p2 = probe_map.at({0, 2}).handle;

    // Expect initial probe values to be the resting potential
    // for the voltage probes (cell membrane potential should
    // be constant), and zero for the current probe.

    auto& state = *(lcell.*fvm_state_ptr).get();
    auto& voltage = state.voltage;

    auto resting = voltage[0];
    EXPECT_NE(0.0, resting);

    // (Probe handles are just pointers in this implementation).
    EXPECT_EQ(resting, *p0);
    EXPECT_EQ(resting, *p1);
    EXPECT_EQ(0.0, *p2);

    // After an integration step, expect voltage probe values
    // to differ from resting, and between each other, and
    // for there to be a non-zero current.
    //
    // First probe, at (0,0), should match voltage in first
    // compartment.

    lcell.integrate(0.01, 0.0025, {}, {});

    EXPECT_NE(resting, *p0);
    EXPECT_NE(resting, *p1);
    EXPECT_NE(*p0, *p1);
    EXPECT_NE(0.0, *p2);

    EXPECT_EQ(voltage[0], *p0);
}

TEST(probe, fvm_lowered_cell_gj) {
    execution_context context;
    std::vector<mc_cell> cells;

    mc_cell l = make_cell_ball_and_stick(false);
    mc_cell c = make_cell_ball_and_stick(false);
    mc_cell r = make_cell_ball_and_stick(false);

    l.add_gap_junction(0, {1, 1}, 1, {1,1}, 0.03);
    c.add_gap_junction(1, {1, 1}, 0, {1,1}, 0.03);

    r.add_gap_junction(2, {1, 1}, 1, {1,1}, 0.03);
    c.add_gap_junction(1, {1, 1}, 2, {1,1}, 0.03);

    i_clamp stim(0, 100, 0.3);
    c.add_stimulus({1, 1}, stim);

    cells.push_back(std::move(l));
    cells.push_back(std::move(c));
    cells.push_back(std::move(r));

    cable1d_recipe rec(cells);

    segment_location loc0{1, 1};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});
    rec.add_probe(2, 20, cell_probe_address{loc0, cell_probe_address::membrane_voltage});

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0, 1, 2}, rec, targets, probe_map);

    EXPECT_EQ(1u, rec.num_probes(0));
    EXPECT_EQ(1u, rec.num_probes(2));
    EXPECT_EQ(2u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({2, 0}).tag);

    probe_handle p_l = probe_map.at({0, 0}).handle;
    probe_handle p_r = probe_map.at({2, 0}).handle;

    // Expect initial probe values to be the resting potential
    // for the voltage probes (cell membrane potential should
    // be constant), and zero for the current probe.

    auto& state = *(lcell.*fvm_state_ptr).get();
    auto& voltage = state.voltage;

    auto resting = voltage[0];
    EXPECT_NE(0.0, resting);

    // (Probe handles are just pointers in this implementation).

    EXPECT_DOUBLE_EQ(resting, *p_l);
    EXPECT_DOUBLE_EQ(resting, *p_r);

    // After an integration step, expect voltage probe values
    // to differ from resting, and between each other, and
    // for there to be a non-zero current.
    //
    // First probe, at (0,0), should match voltage in first
    // compartment.

    lcell.integrate(0.01, 0.0025, {}, {});

    EXPECT_NE(resting, *p_l);
    EXPECT_NE(resting, *p_r);
    EXPECT_DOUBLE_EQ(*p_l, *p_r);
}

TEST(probe, fvm_lowered_cell_gj2) {
    execution_context context;
    std::vector<mc_cell> cells;

    mc_cell l = make_cell_ball_and_stick(false);
    mc_cell c = make_cell_ball_and_stick(false);

    l.add_gap_junction(0, {1, 1}, 1, {1,1}, 0.007);
    c.add_gap_junction(1, {1, 1}, 0, {1,1}, 0.007);

    i_clamp stim(0, 1, 0.3);
    c.add_stimulus({0, 1}, stim);

    cells.push_back(std::move(l));
    cells.push_back(std::move(c));

    cable1d_recipe rec(cells);

    segment_location loc0{1, 1};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0, 1}, rec, targets, probe_map);

    EXPECT_EQ(1u, rec.num_probes(0));
    EXPECT_EQ(1u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);

    lcell.integrate(40, 0.025, {}, {});
}

TEST(probe, fvm_lowered_cell_gj2_1) {
    execution_context context;
    std::vector<mc_cell> cells;

    mc_cell l = make_cell_soma_only(false);
    mc_cell c = make_cell_soma_only(false);

    l.add_gap_junction(0, {0, 1}, 1, {0,1}, 0.002);
    c.add_gap_junction(1, {0, 1}, 0, {0,1}, 0.002);

    i_clamp stim(0, 1, 0.3);
    c.add_stimulus({0, 1}, stim);

    cells.push_back(std::move(l));
    cells.push_back(std::move(c));

    cable1d_recipe rec(cells);

    segment_location loc0{0, 1};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0, 1}, rec, targets, probe_map);

    EXPECT_EQ(1u, rec.num_probes(0));
    EXPECT_EQ(1u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);

    lcell.integrate(40, 0.025, {}, {});
}

TEST(probe, fvm_lowered_cell_gj3) {
    execution_context context;
    std::vector<mc_cell> cells;

    mc_cell c0 = make_cell_ball_and_stick(false);
    mc_cell c1 = make_cell_ball_and_stick(false);
    mc_cell c2 = make_cell_ball_and_stick(false);

    // Three gap junctions
    c0.add_gap_junction(0, {0, 1}, 1, {1, 1}, 0.003);
    c1.add_gap_junction(1, {1, 1}, 0, {0, 1}, 0.003);

    c1.add_gap_junction(1, {0, 1}, 2, {1, 1}, 0.003);
    c2.add_gap_junction(2, {1, 1}, 1, {0, 1}, 0.003);

    c2.add_gap_junction(2, {0, 1}, 0, {1, 1}, 0.003);
    c0.add_gap_junction(0, {1, 1}, 2, {0, 1}, 0.003);

    i_clamp stim(0, 1, 0.3);
    c0.add_stimulus({0, 1}, stim);

    cells.push_back(std::move(c0));
    cells.push_back(std::move(c1));
    cells.push_back(std::move(c2));

    cable1d_recipe rec(cells);

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0, 1, 2}, rec, targets, probe_map);

    lcell.integrate(40, 0.05, {}, {});
}

TEST(probe, fvm_lowered_cell_gj3_1) {
    execution_context context;
    std::vector<mc_cell> cells;

    mc_cell c0 = make_cell_soma_only(false);
    mc_cell c1 = make_cell_soma_only(false);
    mc_cell c2 = make_cell_soma_only(false);

    // Three gap junctions
    c0.add_gap_junction(0, {0, 1}, 1, {0, 1}, 0.003);
    c1.add_gap_junction(1, {0, 1}, 0, {0, 1}, 0.003);

    c1.add_gap_junction(1, {0, 1}, 2, {0, 1}, 0.003);
    c2.add_gap_junction(2, {0, 1}, 1, {0, 1}, 0.003);

    c2.add_gap_junction(2, {0, 1}, 0, {0, 1}, 0.003);
    c0.add_gap_junction(0, {0, 1}, 2, {0, 1}, 0.003);

    i_clamp stim(0, 1, 0.3);
    c0.add_stimulus({0, 1}, stim);

    cells.push_back(std::move(c0));
    cells.push_back(std::move(c1));
    cells.push_back(std::move(c2));

    cable1d_recipe rec(cells);

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0, 1, 2}, rec, targets, probe_map);

    lcell.integrate(40, 0.025, {}, {});
}