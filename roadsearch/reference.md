## The problem

We want the road between two points that costs least: to build, and optionally to drive for years afterward. Any candidate road has a cost, the sum of what each of its meters costs. The difficulty is the number of candidates. Even on a coarse grid there are far more possible roads than could ever be tried one by one.

The way through is an old one: turn the map into a **network** of short, buildable pieces of road, give each piece a cost, and use a **shortest-path algorithm** to find the cheapest chain of pieces from start to goal. These algorithms find the cheapest chain exactly, without trying every chain. Everything below is that idea, made to fit roads.

The steps:

1. Turn the map into a network: where a road can be, and which short pieces join those places.
2. Price each piece: earthwork from the ground's cross slope, plus truck costs if wanted.
3. Find the cheapest chain with Dijkstra's algorithm, or with A\*, which gives the same answer faster.
4. Run Dijkstra from both ends to price a route through every cell, which gives the corridor and the alternatives.
5. Cost the finished routes exactly, with spoil reuse, and rank them.

## 1. The map as a network

The DEM is cut into square cells, 10 m across for the haul road. A road passing through a cell is also pointing somewhere, and it cannot turn on the spot. So a place in the network is a **state**: a cell *and* a heading, one of 32 directions 11.25° apart. The fine grid here has about 260,000 cells, so about 8 million states.

States are joined by **moves**, short pieces of road a real road could follow:

[[figure:lattice]]

| Move | What it is |
| --- | --- |
| Straight | A straight step along the heading |
| Turn | An arc turning 1 or 2 headings, at a radius of at least `R_min` |
| Binned straight | A straight step to any cell within 7 cells, at a direction between two headings (kinks of at most 10° at the ends) |
| Hairpin | A 180° turn: a semicircle, or a keyhole whose legs run 8 m apart while the loop is wider |
| Bend | A 90° or 135° turn that may leave the ground to hold its grade |
| Fill crossing | A straight embankment over a channel, up to `L_cross_max` long |

A move exists only where a road could be built along it. Every point on it must be on open ground, and its grade must be at most `g_max`.

The less obvious moves are there for steep ground:
- **Binned straights** let the road climb at nearly the grade limit in any direction. With exact headings only, the best climb on a 0.25 slope would be 0.061; with them it is 0.098 against a 0.10 limit.
- **Hairpins and bends** let the road turn back on ground steeper than its grade limit, where an ordinary curve through the fall line would be too steep. They carry their own vertical design and pay for the earthwork that needs.

Any road from start to goal is now a chain of moves, and its cost is the sum of their costs. Finding the cheapest road means finding the cheapest chain.

## 2. What a move costs

A move's cost is the cost per meter of the road along it, added up every half cell. The cost per meter comes from the road's **cross-section** at that point, which depends on how steeply the ground falls away across the road: the sidehill slope.

- **Half bench.** The uphill part of the width is cut, and the spoil is cast downhill to build the rest. The split is solved so that cut and fill balance once the spoil has bulked.
- **Full bench.** Where the fill would be too thin, or the ground too steep to hold it, the whole width is cut and the spoil is surplus.
- **Soil and rock.** The cut is split into soil and rock by the soil depth, which thins with slope: d = d0 · exp(−S / S0). Each is priced per m³. Cut faces stand at the soil and rock backslopes and stop at `catch_max`.

The ground is read from a DEM smoothed at the road's width, so slopes are those across the road's footprint, not of single cells. Drainage comes from D8 flow accumulation: cells draining at least `A_min` hectares are channels. Crossing one at grade costs a culvert sized by its drainage area. Ground too steep for any road is excluded, widened by the longest cut that could reach into it.

The cost per meter of road is:

- **Construction:** grading and surfacing (`C_base`), plus cut in soil and rock, plus culverts. Hairpins, bends and fill crossings carry their whole footprint's earthwork, plus any retaining walls.
- **Operating:** if switched on, `C_travel` per meter of road and `C_vertical` per meter of rise or fall. That is the truck costs over the road's life; maintenance, tires, brakes and fleet size are assumed to scale with them, through the weight k.

Units are relative: excavating 1 m³ of soil costs 1. Only ratios change the route.

One rule matters for what follows: **every move costs the same in both directions.** The cost of a piece of road is computed from one fixed end, whichever way it is traveled, so a road from goal to start costs exactly what it costs from start to goal.

## 3. Finding the cheapest chain: Dijkstra

Dijkstra's algorithm grows outward from the start in order of cost. It keeps, for every state, the cheapest cost found so far to reach it, and a list of states not yet **settled**.

1. The start costs 0. Everything else costs "not yet reached".
2. Take the unsettled state with the lowest cost so far, and settle it: its cost is now final.
3. For every move out of it, if the move gives a neighbor a lower cost than it had, lower that neighbor's cost and remember the move.
4. Repeat from 2 until the goal is settled. Then follow the remembered moves back to the start.

[[figure:dijkstra]]

**Why a settled cost is final.** Costs are never negative. Any other way to reach a state would have to leave the settled region through some unsettled state, and every unsettled state already costs at least as much. So no cheaper way exists.

**What it gives.** The cheapest chain on this network, exactly: not an approximation, and the same every time. Ties are broken by state number, so identical runs give identical output. Along the way it also leaves a **cost-to-come** for every state it settled: the cost of the cheapest road from the start to there. Settling in order of cost makes the settled region grow like a rising flood. Its edge, the frontier, is a contour of equal cost, which is what "Watch a search" shows.

## 4. The same answer with less work: A\*

Dijkstra spreads in every direction, including away from the goal. A\* orders the list by **cost so far plus a guaranteed underestimate of the cost still to go**. States that head toward the goal come off the list sooner, and states heading away come off later or never.

The underestimate comes from two facts about any road from a state to the goal:
- it is at least as long as the straight line to the goal;
- it is at least as long as the climb divided by `g_max`.

Every meter costs at least `C_base` + `C_travel`, and every meter of climb costs at least `C_vertical`. So the longer of those two lengths, times the cheapest possible rate, can never exceed the real cost to go.

**Why it is still exact.** The estimate never overshoots, and it changes along any move by no more than that move costs. So when the goal comes off the list, no cheaper route can still be waiting. A\* returns exactly Dijkstra's route and cost; the tests check this on random terrain. It settles about a fifth to a quarter of the cells.

**A\* and Dijkstra are not alternatives here.** A\* is used wherever only the single best route is needed: the fill-price repeats (section 6), the self-overlap re-searches, and the local checks on alternatives. It cannot give the costs of other routes, because most states are never settled.

## 5. The cost of going through anywhere: Dijkstra from both ends

To ask "what is the cheapest road through *this* cell?", every cell needs two numbers:
- the cost of the cheapest road from the start to it;
- the cost of the cheapest road from it to the goal.

A full Dijkstra from the start, not stopped at the goal, gives the first for every cell. A full Dijkstra from the goal gives the second, because every move costs the same both ways.

Their sum, with the heading reversed between the two, is the cost of the cheapest road forced through that cell. Over the whole map, that sum is the **corridor field**.

[[figure:corridor]]

Its lowest value, found along the whole best route, is the best route's cost. Cells close to that value form the near-optimal corridor, and separate low valleys are separate corridors.

These are two complete searches, one from each end. They are not the textbook bidirectional search that stops where two fronts meet. The fronts do first touch at half the best cost, on the best route, which is why the animation marks it.

The corridor field drives:
- **coarse-to-fine clipping:** the coarse pass's corridor decides where the fine pass searches;
- **alternatives:** next section;
- **"Best route through a point"** in the Draw tab.

## 6. Alternatives, and from network cost to true cost

Alternatives are read from the corridor field, one at a time:

1. Take the cell with the lowest corridor cost that is at least `D_alt` (100 m) from every route already chosen and within `eps_alt` (25%) of the best cost.
2. Build the route through it from the two searches.
3. Reject it if it shares more than half its length with a chosen route, or if it overlaps itself.
4. Reject it if it is not locally optimal. Within a window either side of the via cell, its path must cost at most 2% more than the cheapest path between the window ends, found with A\*. This throws out routes that copy the best route and detour out to the via cell and back.
5. Stop at `K_alt` (5) routes or 1,000 candidates.

A prefilter skips via cells whose route would mostly follow the best route, worked out for every cell at once from the two search trees.

The network prices crossing fill at one flat rate per m³. Each finished route is then costed exactly, and routes are ranked by that **true cost**:

- **Mass haul.** Surplus spoil from full-bench cuts and hairpins can fill crossing embankments. A small transport problem assigns spoil to fills by station, with haul priced per m³·m, more for loaded uphill hauls, and nothing beyond 500 m. Fill nobody can supply is bought at `C_borrow`.
- **Fill price iteration.** The first search prices crossing fill at `C_borrow`. The realized cost per m³ on that route becomes the price for the next search. This repeats until the route stops changing, at most five times.
- **Exact sections.** The search reads section costs from a precomputed table. The true cost re-evaluates every segment with the exact section function. The two agree to about 0.01%.
- **Self-overlap.** The best route is checked for a loop back over its own road, which can be cheaper than a hairpin next to flat ground. If found, the excursion is blocked and the search repeated, up to five times.
- **Stacked legs.** Where a road doubles back, the upper leg's fill slope can reach the lower leg's cut. Within a hairpin, overlapping slopes are replaced by a retaining wall priced at `C_wall` per m² of face. Between separate pieces of route, conflicts are reported but not resolved.

## How a run fits together

A run works coarse to fine. A coarse pass (15 m cells) searches the endpoint box plus a 2 km margin. The fine pass (10 m) is limited to the coarse pass's near-optimal corridor plus a buffer, so switchbacks well outside the endpoint box stay possible. If the fine route touches the edge of its region, the buffer is doubled and the pass repeated.

[[figure:pipeline]]

Besides the fill-price repeats inside the A\* step, two loops send the run back:
- a self-overlapping route returns to the search with the loop blocked;
- a route that reaches the edge of its region widens the corridor.

## Burning a route into the DEM

A saved route can be turned into a design surface: the terrain with the road built, plus a cut/fill depth layer. This is typically at 2 m cells within 150 m of the route.

Each cell takes its nearest point on the centerline and that point's design elevation:
- On the road, the cell takes the road's elevation.
- Outside the road edge, cut faces rise at the rock and soil backslopes, and fill slopes fall at the spoil's angle, until they meet the ground.
- Ordinary sections are half benches about the centerline. Hairpins, bends and fill crossings keep their designed profiles.

On a plane, burned cut and fill match the section model within 4%. On the haul route here, burned cut came within 7% of the model and fill 17% above it. The model reads slope from ground smoothed over the road's width, and fill grows sharply with slope, so real ground needs more fill than its average slope suggests.

## Reading the visualization

**Watch a search** replays a finished run from its saved cost fields. No recording of the search is needed: a cell is reached at the moment the search's frontier passes its cost.

| Stage | What it shows | Where to look |
| --- | --- | --- |
| A\* to the goal | Cells settled by A\*, the bright edge being the current frontier | How narrowly A\* heads for the goal |
| Two-way Dijkstra | Fronts from the start (orange) and the goal (teal); white where both have reached | The fronts touch at half the best route's cost, on the best route |
| Corridor and routes | Each cell colored by the cheapest route through it, from the best cost (red) to twice that (pale); the white line marks +25% | Distinct corridors as separate red valleys; each route appears as the threshold passes its cost |

Pale gray ground was never searched: it lies inside the fine grid's rectangle but outside the corridor the coarse pass handed on. A run's Dijkstra stage is shown up to 1.3 times the best cost. Its A\* stage starts at the start cell's lower bound, so the interesting part fills the playback.

**Draw a route** costs a line you draw with the same section model, at the same sample spacing, and sets it against the optimum. Corners are rounded to `R_min`. Nothing is refused: steep grades, excluded ground, side slopes too steep to build and tight corners are flagged instead. A drawn switchback is costed as plain road, so it shows as too steep where the optimizer would build a hairpin.

**Best route through a point**, in the Draw tab, reads the corridor field for the cost of the cheapest route through any cell, and draws that route from the saved search trees.

**Compare runs** puts the best route of every run on one map. Each run weights truck costs by a factor k: k = 1 is one 385 t truck for five years at the given fuel figures, and k = 0 is construction only. Rows give construction and operating cost, length and total rise plus fall.

## Simplifications and key parameters

The route is optimal on the search's network of states and moves, for the costs as modeled. The model leaves out the following:

- Road elevation follows the ground except through hairpins, bends and fill crossings. There are no vertical curves and no through-cuts.
- Sections assume locally planar ground and ignore the curvature of the road in plan. Straight moves may meet at kinks of up to 10°.
- Soil depth depends on slope only; there is no geology map.
- Stacked-leg conflicts between separate pieces of route are reported, not resolved.
- Truck costs per meter and per meter of rise or fall come from fuel figures. Everything else is assumed to scale with them, and there is no time cost.
- A diagonal move can pass between the cells of a diagonal channel without paying for a culvert.
- Construction and wall costs are placeholders.

| Parameter | Haul-road value | Meaning |
| --- | --- | --- |
| `w` | 20 m | road width |
| `g_max` | 8% | maximum grade |
| `R_min` | 40 m | minimum curve radius; also the keyhole reverse curves |
| `R_hp`, `w_hp`, `g_hp` | 25 m, 25 m, 3% | hairpin loop radius, width and centerline grade |
| `hp_gap_min` | 8 m | clear gap between keyhole legs |
| `C_base` | 40 per m | grading and surfacing (unconfirmed) |
| `C_wall` | 15 per m² | retaining wall face (unconfirmed) |
| `C_travel`, `C_vertical` | 125.5 per m, 2,576 per m of rise or fall | truck costs at k = 1: one truck for five years, $4 per unit |
| `move_max_cells`, `kink_max_deg` | 7, 10° | binned straight moves |
| `N_theta` | 32 | headings |
| `eps_alt`, `D_alt`, `K_alt` | 25%, 100 m, 5 | alternatives |
