/*
 * A miniapp that demonstrates using an external spike source.
 * Actual miniapp that runs real arbor -- connect to real nest or nest proxy
 */

#include <arbor/version.hpp>

#ifndef ARB_MPI_ENABLED

#include <iostream>

int main() {
    std::cerr << "**** Only runs with ARB_MPI_ENABLED ***" << std::endl;
    return 1;
}

#else //ARB_MPI_ENABLED

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/swcio.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <sup/json_params.hpp>

#include "branch_cell.hpp"

#include <mpi.h>
#include <arborenv/with_mpi.hpp>

#include "parameters.hpp"
#include "mpiutil.hpp"
#include "branch_cell.hpp"

#include <sup/tinyopt.hpp>

struct options {
    std::string swc_file;
    double t_end = 20;
    double dt = 0.025;
    float syn_weight = 0.01;
};

options parse_options(int argc, char** argv);
arb::morphology default_morphology();
arb::morphology read_swc(const std::string& path);


using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

class ring_recipe: public arb::recipe {
public:
    ring_recipe(unsigned num_cells, cell_parameters params, double min_delay, int num_nest_cells):
        num_cells_(num_cells),
        cell_params_(params),
        min_delay_(min_delay),
        num_nest_cells_(num_nest_cells)
    {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return branch_cell(gid, cell_params_);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override {
        return 1;
    }

    // Each cell has one incoming connection from an external source.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        cell_gid_type src = num_cells_ + (gid%num_nest_cells_); // round robin
        return {arb::cell_connection({src, 0}, {gid, 0}, event_weight_, min_delay_)};
    }

    // No event generators.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        return {};
    }

    // There is one probe (for measuring voltage at the soma) on the cell.
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage.
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma.
        arb::mlocation loc{0, 0.0};

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

    arb::util::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }

private:
    cell_size_type num_cells_;
    cell_parameters cell_params_;
    double min_delay_;
    float event_weight_ = 0.01;
    int num_nest_cells_;
    arb::cable_cell_global_properties gprop_;
};


struct single_recipe: public arb::recipe {
  explicit single_recipe(arb::morphology m, size_t num_cells, size_t num_nest_cells): morpho(std::move(m)),
										      num_cells_(num_cells), num_nest_cells_(num_nest_cells){
        gprop.default_parameters = arb::neuron_parameter_defaults;
    }

    arb::cell_size_type num_cells() const override { return num_cells_; }
    arb::cell_size_type num_probes(arb::cell_gid_type) const override { return 1; }
    arb::cell_size_type num_targets(arb::cell_gid_type) const override { return 1; }

    arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        arb::mlocation mid_soma = {0, 0.5};
        arb::cell_probe_address probe = {mid_soma, arb::cell_probe_address::membrane_voltage};

        // Probe info consists of: the probe id, a tag value to distinguish this probe
        // from others for any attached sampler (unused), and the cell probe address.

        return {probe_id, 0, probe};
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    arb::util::any get_global_properties(arb::cell_kind) const override {
        return gprop;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        arb::label_dict dict;
        using arb::reg::tagged;
        dict.set("soma", tagged(1));
        dict.set("dend", join(tagged(3), tagged(4), tagged(42)));
        arb::cable_cell c(morpho, dict, false);

        // Add HH mechanism to soma, passive channels to dendrites.
        c.paint("soma", "hh");
        c.paint("dend", "pas");

        // Discretize dendrites according to the NEURON d-lambda rule.
        /* skip this during refactoring of the morphology interface
        for (std::size_t i=1; i<c.num_branches(); ++i) {
            arb::cable_segment* branch = c.cable(i);

            double dx = c.segment_length_constant(100., i, gprop.default_parameters)*0.3;
            unsigned n = std::ceil(branch->length()/dx);
            branch->set_compartments(n);
        }
        */

        // Add synapse to last branch.

        arb::cell_lid_type last_branch = c.num_branches()-1;
        arb::mlocation end_last_branch = { last_branch, 1. };
        c.place(end_last_branch, "exp2syn");

        return c;
    }

    // Each cell has one incoming connection from an external source.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        cell_gid_type src = num_cells_ + (gid%num_nest_cells_); // round robin
        return {arb::cell_connection({src, 0}, {gid, 15}, 1.0, 1.0)}; //event_weight_, min_delay_)};
    }

    arb::morphology morpho;
    arb::cable_cell_global_properties gprop;
  size_t num_cells_;
  size_t num_nest_cells_;
};




struct cell_stats {
    using size_type = unsigned;
    cell_size_type ncells = 0;
    size_type nsegs = 0;
    size_type ncomp = 0;

    cell_stats(arb::recipe& r, comm_info& info) {
        int nranks, rank;
        MPI_Comm_rank(info.comm, &rank);
        MPI_Comm_size(info.comm, &nranks);
        ncells = r.num_cells();
        size_type cells_per_rank = ncells/nranks;
        size_type b = rank*cells_per_rank;
        size_type e = (rank==nranks-1)? ncells: (rank+1)*cells_per_rank;
        size_type nsegs_tmp = 0;
        size_type ncomp_tmp = 0;
        for (size_type i=b; i<e; ++i) {
            auto c = arb::util::any_cast<arb::cable_cell>(r.get_cell_description(i));
            nsegs_tmp += c.segments().size();
            ncomp_tmp += c.num_compartments();
        }
        MPI_Allreduce(&nsegs_tmp, &nsegs, 1, MPI_UNSIGNED, MPI_SUM, info.comm);
        MPI_Allreduce(&ncomp_tmp, &ncomp, 1, MPI_UNSIGNED, MPI_SUM, info.comm);
    }

    friend std::ostream& operator<<(std::ostream& o, const cell_stats& s) {
        return o << "cell stats: "
                 << s.ncells << " cells; "
                 << s.nsegs << " segments; "
                 << s.ncomp << " compartments.";
    }
};

// callback for external spikes
struct extern_callback {
    comm_info info;

    extern_callback(comm_info info): info(info) {}

    std::vector<arb::spike> operator()(arb::time_type t) {
        std::vector<arb::spike> local_spikes; // arbor processes send no spikes
        print_vec_comm("ARB-send", local_spikes, info.comm);
        static int step = 0;
        std::cerr << "ARB: step " << step++ << std::endl;
        auto global_spikes = gather_spikes(local_spikes, MPI_COMM_WORLD);
        print_vec_comm("ARB-recv", global_spikes, info.comm);

        return global_spikes;
    }
};

//
//  N ranks = Nn + Na
//      Nn = number of nest ranks
//      Na = number of arbor ranks
//
//  Nest  on COMM_WORLD [0, Nn)
//  Arbor on COMM_WORLD [Nn, N)
//

int main(int argc, char** argv) {

    options opt = parse_options(argc, argv);

    try {
        arbenv::with_mpi guard(argc, argv, false);

        auto info = get_comm_info(true);

        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }
        resources.gpu_id = arbenv::find_private_gpu(info.comm);
        auto context = arb::make_context(resources, info.comm);
        
        const bool root = arb::rank(context) == 0;
        std::cout << sup::mask_stream(root);

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(context) << "\n";
        std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        auto params = read_options(argc, argv);
        std::cout << "ARB: Params: " << params << std::endl;


#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(context);
#endif

        arb::profile::meter_manager meters;
        meters.start(context);

        std::cout << "ARB: starting handshake" << std::endl;

        // hand shake #1: communicate cell populations
        broadcast((int)params.num_cells, MPI_COMM_WORLD, info.arbor_root);
        int num_nest_cells = broadcast(0,  MPI_COMM_WORLD, info.nest_root);

        std::cout << "ARB: num_nest_cells: " << num_nest_cells << std::endl;

        // Create an instance of our recipe.
        // ring_recipe recipe(params.num_cells, params.cell, params.min_delay, num_nest_cells);
	single_recipe recipe(read_swc("../example/single/example.swc"), //opt.swc_file),  // opt.swc_file.empty()? default_morphology(): 
			     params.num_cells, num_nest_cells);
        cell_stats stats(recipe, info);
        std::cout << stats << std::endl;

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // hand shake #2: min delay
        float arb_comm_time = sim.min_delay()/2;
        std::cout << "ARB: arb_comm_time=" << arb_comm_time << std::endl;
        broadcast(arb_comm_time, MPI_COMM_WORLD, info.arbor_root);
        float nest_comm_time = broadcast(0.f, MPI_COMM_WORLD, info.nest_root);
        std::cout << "ARB: nest_comm_time=" << nest_comm_time << std::endl;
        auto min_delay = sim.min_delay(nest_comm_time*2);
        std::cout << "ARB: min_delay=" << min_delay << std::endl;

        float delta = min_delay/2;
        float sim_duration = params.duration;
        unsigned steps = sim_duration/delta;
        if (steps*delta < sim_duration) ++steps;

        //hand shake #3: steps
        broadcast(steps, MPI_COMM_WORLD, info.arbor_root);

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (root) {
            sim.set_global_spike_callback(
                [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                    print_vec_comm("ARB", spikes);
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        }

        // Define the external spike source callback
        sim.set_external_spike_callback(extern_callback(info));

        meters.checkpoint("model-init", context);

        std::cout << "ARB: running simulation" << std::endl;
        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.duration, 0.025);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        if (root) {
            std::cout << "\nARB: " << ns << " spikes generated at rate of "
                      << params.duration/ns << " ms between spikes\n";
            std::ofstream fid("spikes.gdf");
            if (!fid.good()) {
                std::cerr << "ARB: Warning: unable to open file spikes.gdf for spike output\n";
            }
            else {
                char linebuf[45];
                for (auto spike: recorded_spikes) {
                    auto n = std::snprintf(
                        linebuf, sizeof(linebuf), "%u %.4f\n",
                        unsigned{spike.source.gid}, float(spike.time));
                    fid.write(linebuf, n);
                }
            }
        }

        auto report = arb::profile::make_meter_report(meters, context);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }
    
    return 0;
}


options parse_options(int argc, char** argv) {
    using namespace to;
    options opt;

    char** arg = argv+1;
    while (*arg) {
        if (auto dt = parse_opt<double>(arg, 'd', "dt")) {
            opt.dt = dt.value();
        }
        else if (auto t_end = parse_opt<double>(arg, 't', "t-end")) {
            opt.t_end = t_end.value();
        }
        else if (auto weight = parse_opt<float>(arg, 'w', "weight")) {
            opt.syn_weight = weight.value();
        }
        else if (auto swc = parse_opt<std::string>(arg, 'm', "morphology")) {
            opt.swc_file = swc.value();
        }
        else {
            usage(argv[0], "[-m|--morphology SWCFILE] [-d|--dt TIME] [-t|--t-end TIME] [-w|--weight WEIGHT]");
            std::exit(1);
        }
    }
    return opt;
}

// If no SWC file is given, the default morphology consists
// of a soma of radius 6.3 µm and a single unbranched dendrite
// of length 200 µm and radius decreasing linearly from 0.5 µm
// to 0.2 µm.

arb::morphology default_morphology() {
    arb::sample_tree samples;

    auto p = samples.append(arb::msample{{  0.0, 0.0, 0.0, 6.3}, 1});
    p = samples.append(p, arb::msample{{  6.3, 0.0, 0.0, 0.5}, 3});
    p = samples.append(p, arb::msample{{206.3, 0.0, 0.0, 0.2}, 3});

    return arb::morphology(std::move(samples));
}

arb::morphology read_swc(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("unable to open SWC file: "+path);

    return arb::morphology(arb::swc_as_sample_tree(arb::parse_swc_file(f)));
}

#endif //ARB_MPI_ENABLED

