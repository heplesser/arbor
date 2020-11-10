#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <sup/json_params.hpp>

#include "branch_cell.hpp"

struct ring_params {
    ring_params() = default;

    std::string name = "default";
    unsigned num_cells = 100;
    double min_delay = 10;
    double duration = 100;
    cell_parameters cell;
};

std::ostream& operator<<(std::ostream& o, ring_params& p) {
    return o << "name=" << p.name
             << ", duration=" << p.duration
             << ", num_cells=" << p.num_cells
             << ", min_delay=" << p.min_delay
             << ", cell=[" << p.cell << "]";
}

ring_params read_options(int argc, char** argv) {
    using sup::param_from_json;

    ring_params params;
    if (argc<2) {
        std::cout << "Using default parameters.\n";
        return params;
    }
    if (argc>2) {
        throw std::runtime_error("More than command line one option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(params.name, "name", json);
    param_from_json(params.num_cells, "num-cells", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.min_delay, "min-delay", json);
    param_from_json(params.cell.max_depth, "depth", json);
    param_from_json(params.cell.branch_probs, "branch-probs", json);
    param_from_json(params.cell.compartments, "compartments", json);
    param_from_json(params.cell.lengths, "lengths", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}