# Lead QA Spec Audit - 2026-06-19

Scope:

- Client spec: `tmp/pdfs/pilot-ai-lead-data-sourcing/extracted_layout_fresh.txt`
- Audited files: the 10 CSVs named by Ashish in the request
- Total rows audited: 190
- Quantity check: all requested quantities are met exactly
- Schema check: all 10 files parse with the same 19 client-ready columns
- Geography check: 0 hard misses found across ZIP, county, radius, or state filters
- Source URL reachability spot-check: 306 unique route/signal URLs checked by GET; 244 returned HTTP 200, 56 were bot-gated with 403/406, 6 timed out, and 0 were confirmed broken by GET

Strict rubric:

- `10/10` means the row matches the project, allowed industry/property type, geography, and the stated trigger/qualifier with enough row evidence to defend it to the client.
- I did not treat "buying intent not verified" alone as a hard failure, because these are cold-call lead lists. I did treat missing proof for hard qualifiers, weak core-use-case evidence, wrong/adjacent business type, or portfolio-only routing as not 10/10.

Executive result:

- Hard disqualifying leads found: 0
- Clean 10/10 rows: 154
- Not literal 10/10 rows: 36
- Recommended action: do not tell the client every lead is a 10/10. Send the clean rows confidently; verify or replace the 36 rows below depending on how strict the client is about proof.

## Client Summary

| Client | Rows | Clean 10/10 | Not 10/10 | Main issue |
|---|---:|---:|---:|---|
| 1-800 Striper Long Island | 20 | 20 | 0 | No spec-fit misses found |
| All Janitorial Service | 20 | 20 | 0 | No spec-fit misses found |
| Heroes Lawn Care | 20 | 20 | 0 | No spec-fit misses found |
| Archer HVAC & Plumbing | 20 | 19 | 1 | One portfolio route, not a specific facility |
| Firm Facility Services | 20 | 18 | 2 | Two adjacent commercial-property routes |
| Tectum Roofing | 30 | 28 | 2 | Two portfolio/developer routes, not specific roof assets |
| Unique Construction | 20 | 19 | 1 | One national-footprint row is too broad for East Coast focus |
| Absolute Vending Solutions | 10 | 9 | 1 | One row has weaker waiting-room/amenity evidence |
| MSA Worldwide | 10 | 1 | 9 | Most rows use revenue proxies, not direct $400k annual revenue proof |
| Ephraim Solutions | 20 | 0 | 20 | All rows use $5M revenue and AI-interest proxies, not direct proof |

## Not 10/10 Rows

### Absolute Vending Solutions

- Row 10, Stiles Automotive: industry and geography fit, but the row itself says public waiting-room evidence is weaker than the top rows. This is usable, but not a 10/10 until phone-confirmed for customer dwell/amenity need and no existing vending setup.

### MSA Worldwide

Client hard qualifier: minimum $400,000 annual revenue.

- Row 1, Ziggi's Coffee: strong franchise signal, but revenue is supported by unit/investment proxy, not direct annual revenue proof.
- Row 2, Foxtail Coffee Co.: strong franchise signal, but exact revenue is not public in the row.
- Row 3, Just Love Coffee Cafe: strong franchise signal, but exact revenue is not public in the row.
- Row 4, Toastique: strong franchise signal, but exact revenue is not public in the row.
- Row 5, Mighty Dog Roofing: strong franchise signal, but revenue is inferred from franchise scale/performance coverage.
- Row 7, Koala Insulation: strong franchise signal, but revenue is inferred from franchise model/investment.
- Row 8, Painter Bros: investment range is not the same as annual revenue; not a clean financial qualifier match.
- Row 9, All Dry Services: strong franchise signal, but exact revenue is not public in the row.
- Row 10, Garage Living: unit/investment proxy only; not direct annual revenue proof.

Row 6, LIME Painting is the only MSA row with direct row evidence clearing the financial qualifier via official average owner volume.

### Ephraim Solutions

Client hard qualifier: minimum $5,000,000 annual revenue. Client trigger also asks for businesses interested in AI operations integration.

All 20 rows are in the right geography and service vertical, but none directly prove $5M revenue or actual AI buying interest. They are usable as AI-ops call targets, not literal 10/10 qualified leads.

Strong scale-proxy rows:

- Rows 1-6: Milestone; Baker Brothers; Berkeys; A#1 Air; Tempo Air; On Time Experts.
- Rows 10-20: Rescue Air; Lon Smith Roofing; Kidd Roofing; Dalworth Restoration; BMS CAT / Blackmon Mooring; ABC Home & Commercial Services; Massey Services; Southern Botanical; Yellowstone Landscape; Overhead Door Company of Dallas; Levy & Son Service Experts.

Weaker revenue-proof rows to replace first if the client demands strict $5M proof:

- Row 7, Frymire Home Services
- Row 8, Strittmatter Plumbing, Heating & AC
- Row 9, Cody & Sons Plumbing, Heating & Air

### Firm Facility Services

- Row 18, Lincoln Property Company - Dallas: strong Dallas property-management route, but the client business-type list is restaurant/QSR, hospitality, multifamily, retail, or franchisee owners. General commercial office property management is adjacent, not a perfect 10/10.
- Row 19, PegasusAblon: strong route quality, but mixed-use/commercial property management is adjacent unless verified as retail/multifamily/hospitality maintenance work.

### Archer HVAC & Plumbing

- Row 20, FirstService Residential - North Las Vegas: ZIP and service need fit, but it is a portfolio/community-management route rather than a specific hospital, hotel, shopping center, grocery, dealership, office park, gym, or multifamily property. Verify the target property set before treating as 10/10.

### Tectum Roofing

- Row 29, Norwood Development Group: valid commercial developer/landlord route, but the row is not a specific 5,000+ sqft roof asset. Good door-opener, not a perfect property-level lead.
- Row 30, Colorado Springs Commercial - Cushman & Wakefield Alliance: valid commercial property route, but not a specific 5,000+ sqft roof asset. Good portfolio route, not a perfect property-level lead.

### Unique Construction

- Row 12, Tractor Supply Company: strong national retailer and construction-leadership route, but the geography row is broad national footprint/49-state presence rather than a concrete East Coast or Northeast expansion/buildout signal. Keep only if the call script ties to specific target-state projects.

## Clean Files

No not-10/10 rows were found in:

- `1-800_Striper_Long_Island_STANDARDIZED_CALL_TEAM_READY_2026-06-18.csv`
- `All_Janitorial_Service_Redwood_City_STANDARDIZED_CALL_TEAM_READY_2026-06-18.csv`
- `Heroes_Lawn_Care_NJ_STANDARDIZED_CALL_TEAM_READY_2026-06-18.csv`

## Final QA Call

This batch is client-usable, but the honest label is not "190 perfect 10/10 leads." The stronger claim is:

> 190 rows were audited against the client spec. No hard geography or industry disqualifications were found. 154 rows are clean 10/10 spec fits. 36 rows are usable but need verification or replacement before being represented as 10/10.
