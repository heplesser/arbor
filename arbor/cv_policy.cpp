#include <utility>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include "util/rangeutil.hpp"
#include "util/span.hpp"

// Discretization policy implementations:

namespace arb {

static auto unique_sum = [](auto&&... lss) {
    return ls::support(sum(std::forward<decltype(lss)>(lss)...));
};

// Combinators:
// cv_policy_plus_ represents the result of operator+,
// cv_policy_bar_ represents the result of operator|.

struct cv_policy_plus_: cv_policy_base {
    cv_policy_plus_(const cv_policy& lhs, const cv_policy& rhs):
        lhs_(lhs), rhs_(rhs) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_plus_(*this));
    }

    locset cv_boundary_points(const cable_cell& c) const override {
        return unique_sum(lhs_.cv_boundary_points(c), rhs_.cv_boundary_points(c));
    }

    region domain() const override { return join(lhs_.domain(), rhs_.domain()); }

    cv_policy lhs_, rhs_;
};

cv_policy operator+(const cv_policy& lhs, const cv_policy& rhs) {
    return cv_policy_plus_(lhs, rhs);
}

struct cv_policy_bar_: cv_policy_base {
    cv_policy_bar_(const cv_policy& lhs, const cv_policy& rhs):
        lhs_(lhs), rhs_(rhs) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_bar_(*this));
    }

    locset cv_boundary_points(const cable_cell& c) const override {
        return unique_sum(ls::restrict(lhs_.cv_boundary_points(c), complement(rhs_.domain())), rhs_.cv_boundary_points(c));
    }

    region domain() const override { return join(lhs_.domain(), rhs_.domain()); }

    cv_policy lhs_, rhs_;
};

cv_policy operator|(const cv_policy& lhs, const cv_policy& rhs) {
    return cv_policy_bar_(lhs, rhs);
}

// Public policy implementations:

locset cv_policy_explicit::cv_boundary_points(const cable_cell& cell) const {
    return
        ls::support(
            util::foldl(
                [this](locset l, const auto& comp) {
                    return sum(std::move(l), ls::restrict(locs_, comp));
                },
                ls::boundary(domain_),
                components(cell.morphology(), thingify(domain_, cell.provider()))));
}

locset cv_policy_max_extent::cv_boundary_points(const cable_cell& cell) const {
    const unsigned nbranch = cell.morphology().num_branches();
    const auto& embed = cell.embedding();
    if (!nbranch || max_extent_<=0) return ls::nil();

    std::vector<mlocation> points;
    double oomax_extent = 1./max_extent_;
    auto comps = components(cell.morphology(), thingify(domain_, cell.provider()));

    for (auto& comp: comps) {
        for (mcable c: comp) {
            double cable_length = embed.integrate_length(c);
            unsigned ncv = std::ceil(cable_length*oomax_extent);
            double scale = (c.dist_pos-c.prox_pos)/ncv;

            if (flags_&cv_policy_flag::interior_forks) {
                for (unsigned i = 0; i<ncv; ++i) {
                    points.push_back({c.branch, c.prox_pos+(1+2*i)*scale/2});
                }
            }
            else {
                for (unsigned i = 0; i<ncv; ++i) {
                    points.push_back({c.branch, c.prox_pos+i*scale});
                }
                points.push_back({c.branch, c.dist_pos});
            }
        }
    }

    util::sort(points);
    return unique_sum(locset(std::move(points)), ls::cboundary(domain_));
}

locset cv_policy_fixed_per_branch::cv_boundary_points(const cable_cell& cell) const {
    const unsigned nbranch = cell.morphology().num_branches();
    if (!nbranch) return ls::nil();

    std::vector<mlocation> points;
    double ooncv = 1./cv_per_branch_;
    auto comps = components(cell.morphology(), thingify(domain_, cell.provider()));

    for (auto& comp: comps) {
        for (mcable c: comp) {
            double scale = (c.dist_pos-c.prox_pos)*ooncv;

            if (flags_&cv_policy_flag::interior_forks) {
                for (unsigned i = 0; i<cv_per_branch_; ++i) {
                    points.push_back({c.branch, c.prox_pos+(1+2*i)*scale/2});
                }
            }
            else {
                for (unsigned i = 0; i<cv_per_branch_; ++i) {
                    points.push_back({c.branch, c.prox_pos+i*scale});
                }
                points.push_back({c.branch, c.dist_pos});
            }
        }
    }

    util::sort(points);
    return unique_sum(locset(std::move(points)), ls::cboundary(domain_));
}

locset cv_policy_every_segment::cv_boundary_points(const cable_cell& cell) const {
    const unsigned nbranch = cell.morphology().num_branches();
    if (!nbranch) return ls::nil();

    return unique_sum(
                ls::cboundary(domain_),
                ls::restrict(ls::segment_boundaries(), domain_));
}

} // namespace arb
