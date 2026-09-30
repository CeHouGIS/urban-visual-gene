# 512D Semantic Parent–Child Typed Hierarchy (v2)

## Scope

- Analysis unit: complete four-direction panorama
- Cities: 30
- Panoramas: 296,462
- Patches: 232,426,208
- Patch winners: Top-4 identities only (activation magnitudes discarded after ranking)
- Semantic parents with assigned dimensions: 7
- Latent child concepts: 61
- Selected AND dimension pairs: 185
- Selected OR dimension pairs: 4
- Child-level typed edges: 28

## Interpretation

The hierarchy is a tree only from semantic parent to latent child. AND and OR are lateral, typed relations between children under the same parent. AND means stable positive co-composition; OR means avoidance despite similar context and spatial role, consistent with substitution. UNRESOLVED pairs are deliberately not forced into either type.

## Child concepts

- **RDP-C01 — Road / Building**: 24 dimensions; bootstrap stability 0.638; representatives D436; D441; D101; D377; D322; D120.
- **RDP-C02 — Road / Lane Marking - General**: 22 dimensions; bootstrap stability 0.430; representatives D105; D361; D159; D067; D476; D186.
- **RDP-C03 — Road / Lane Marking - General**: 21 dimensions; bootstrap stability 0.510; representatives D335; D031; D236; D438; D038; D106.
- **RDP-C04 — Road / Vegetation**: 8 dimensions; bootstrap stability 0.586; representatives D466; D158; D340; D200; D280; D091.
- **RDP-C05 — Road / Sidewalk**: 7 dimensions; bootstrap stability 0.548; representatives D328; D109; D230; D239; D099; D187.
- **RDP-C06 — Road / Pedestrian Area**: 5 dimensions; bootstrap stability 0.760; representatives D171; D442; D078; D406; D268.
- **RDP-C07 — Sidewalk / Fence**: 3 dimensions; bootstrap stability 0.850; representatives D177; D504; D387.
- **RDP-C08 — Sidewalk / Road**: 2 dimensions; bootstrap stability 0.600; representatives D474; D252.
- **RDP-C09 — Road / Vegetation**: 2 dimensions; bootstrap stability 1.000; representatives D260; D323.
- **RDP-C10 — Sky / Road**: 2 dimensions; bootstrap stability 0.400; representatives D324; D334.
- **RDP-C11 — Road / Sidewalk**: 1 dimensions; bootstrap stability 1.000; representatives D087.
- **RDP-C12 — Sidewalk / Road**: 1 dimensions; bootstrap stability 1.000; representatives D339.
- **BLT-C01 — Building / Vegetation**: 44 dimensions; bootstrap stability 0.541; representatives D503; D168; D138; D292; D278; D458.
- **BLT-C02 — Bridge / Building**: 18 dimensions; bootstrap stability 0.957; representatives D215; D055; D209; D182; D356; D315.
- **BLT-C03 — Building / Fence**: 15 dimensions; bootstrap stability 0.526; representatives D085; D261; D195; D510; D314; D374.
- **BLT-C04 — Building / Vegetation**: 14 dimensions; bootstrap stability 0.405; representatives D319; D114; D089; D348; D392; D069.
- **BLT-C05 — Building / Wall**: 14 dimensions; bootstrap stability 0.317; representatives D430; D465; D413; D303; D223; D224.
- **BLT-C06 — Building / Sky**: 11 dimensions; bootstrap stability 0.686; representatives D349; D005; D483; D262; D075; D360.
- **BLT-C07 — Road / Wall**: 10 dimensions; bootstrap stability 0.370; representatives D491; D163; D045; D116; D460; D480.
- **BLT-C08 — Bridge / Building**: 8 dimensions; bootstrap stability 0.329; representatives D477; D020; D025; D014; D133; D126.
- **BLT-C09 — Fence / Tunnel**: 7 dimensions; bootstrap stability 0.598; representatives D372; D419; D217; D082; D355; D461.
- **BLT-C10 — Wall / Vegetation**: 5 dimensions; bootstrap stability 0.340; representatives D293; D453; D304; D129; D468.
- **BLT-C11 — Barrier / Road**: 5 dimensions; bootstrap stability 0.960; representatives D213; D152; D211; D370; D327.
- **BLT-C12 — Building / Sky**: 4 dimensions; bootstrap stability 0.675; representatives D277; D065; D097; D493.
- **BLT-C13 — Tunnel / Bridge**: 4 dimensions; bootstrap stability 1.000; representatives D285; D424; D489; D411.
- **BLT-C14 — Fence / Building**: 2 dimensions; bootstrap stability 1.000; representatives D249; D227.
- **BLT-C15 — Vegetation / Building**: 2 dimensions; bootstrap stability 0.900; representatives D497; D484.
- **BLT-C16 — Bridge / Vegetation**: 1 dimensions; bootstrap stability 1.000; representatives D254.
- **BLT-C17 — Bridge / Building**: 1 dimensions; bootstrap stability 1.000; representatives D402.
- **BLT-C18 — Fence / Road**: 1 dimensions; bootstrap stability 1.000; representatives D475.
- **TRN-C01 — Terrain / Vegetation**: 11 dimensions; bootstrap stability 0.864; representatives D499; D471; D404; D170; D382; D508.
- **TRN-C02 — Terrain / Sky**: 10 dimensions; bootstrap stability 0.742; representatives D449; D454; D048; D455; D216; D173.
- **TRN-C03 — Road / Sand**: 3 dimensions; bootstrap stability 0.700; representatives D403; D391; D291.
- **TRN-C04 — Mountain / Vegetation**: 1 dimensions; bootstrap stability 1.000; representatives D241.
- **SKY-C01 — Sky / Vegetation**: 19 dimensions; bootstrap stability 0.542; representatives D269; D131; D245; D350; D353; D342.
- **SKY-C02 — Sky / Building**: 15 dimensions; bootstrap stability 0.684; representatives D113; D157; D399; D250; D098; D136.
- **SKY-C03 — Sky / Building**: 13 dimensions; bootstrap stability 0.794; representatives D161; D180; D284; D300; D364; D181.
- **SKY-C04 — Sky / Vegetation**: 10 dimensions; bootstrap stability 0.643; representatives D270; D437; D282; D310; D197; D421.
- **SKY-C05 — Sky / Building**: 6 dimensions; bootstrap stability 0.577; representatives D086; D256; D169; D094; D189; D108.
- **SKY-C06 — Sky / Vegetation**: 5 dimensions; bootstrap stability 0.750; representatives D479; D084; D375; D225; D258.
- **SKY-C07 — Sky / Vegetation**: 2 dimensions; bootstrap stability 1.000; representatives D380; D247.
- **SKY-C08 — Sky / Building**: 1 dimensions; bootstrap stability 1.000; representatives D134.
- **VEG-C01 — Vegetation / Building**: 26 dimensions; bootstrap stability 0.456; representatives D210; D144; D433; D233; D290; D167.
- **VEG-C02 — Vegetation / Terrain**: 22 dimensions; bootstrap stability 0.726; representatives D124; D093; D221; D103; D450; D047.
- **VEG-C03 — Vegetation / Building**: 19 dimensions; bootstrap stability 0.635; representatives D119; D451; D429; D002; D511; D318.
- **VEG-C04 — Vegetation / Building**: 12 dimensions; bootstrap stability 0.452; representatives D052; D265; D496; D398; D027; D081.
- **VEG-C05 — Vegetation / Sky**: 10 dimensions; bootstrap stability 0.423; representatives D434; D140; D457; D275; D196; D155.
- **VEG-C06 — Vegetation / Building**: 6 dimensions; bootstrap stability 0.570; representatives D001; D439; D423; D296; D238; D039.
- **VEG-C07 — Vegetation / Building**: 5 dimensions; bootstrap stability 0.420; representatives D015; D363; D059; D146; D371.
- **VEG-C08 — Vegetation / Road**: 4 dimensions; bootstrap stability 0.367; representatives D164; D077; D257; D068.
- **VEG-C09 — Vegetation / Road**: 2 dimensions; bootstrap stability 1.000; representatives D358; D273.
- **FUR-C01 — Vegetation / Building**: 3 dimensions; bootstrap stability 1.000; representatives D279; D112; D389.
- **FUR-C02 — Truck / Billboard**: 1 dimensions; bootstrap stability 1.000; representatives D388.
- **VEH-C01 — Car / Sky**: 10 dimensions; bootstrap stability 0.922; representatives D142; D267; D049; D204; D016; D115.
- **VEH-C02 — Car / Ego Vehicle**: 7 dimensions; bootstrap stability 0.843; representatives D329; D271; D042; D007; D330; D352.
- **VEH-C03 — Truck / Road**: 7 dimensions; bootstrap stability 0.986; representatives D151; D090; D096; D435; D058; D286.
- **VEH-C04 — Vegetation / Car**: 5 dimensions; bootstrap stability 0.645; representatives D313; D147; D040; D125; D321.
- **VEH-C05 — Truck / Building**: 4 dimensions; bootstrap stability 0.583; representatives D326; D467; D018; D166.
- **VEH-C06 — Road / Ego Vehicle**: 4 dimensions; bootstrap stability 0.900; representatives D092; D431; D381; D026.
- **VEH-C07 — Bus / Building**: 3 dimensions; bootstrap stability 0.933; representatives D464; D390; D043.
- **VEH-C08 — Motorcycle / Building**: 2 dimensions; bootstrap stability 0.600; representatives D214; D246.

## Typed child relations

- **AND** VEH-C01 ↔ VEH-C02: score 3.584, 1 supporting pair(s), city sign fraction 1.000.
- **AND** RDP-C01 ↔ RDP-C03: score 3.189, 1 supporting pair(s), city sign fraction 0.889.
- **AND** SKY-C01 ↔ SKY-C03: score 3.163, 1 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C01 ↔ BLT-C02: score 3.021, 6 supporting pair(s), city sign fraction 0.981.
- **AND** RDP-C02 ↔ RDP-C03: score 2.778, 2 supporting pair(s), city sign fraction 0.979.
- **AND** VEG-C03 ↔ VEG-C05: score 2.752, 2 supporting pair(s), city sign fraction 1.000.
- **AND** VEG-C01 ↔ VEG-C03: score 2.632, 8 supporting pair(s), city sign fraction 0.984.
- **AND** VEG-C03 ↔ VEG-C04: score 2.632, 3 supporting pair(s), city sign fraction 0.983.
- **AND** VEH-C03 ↔ VEH-C05: score 2.593, 1 supporting pair(s), city sign fraction 1.000.
- **AND** VEG-C01 ↔ VEG-C04: score 2.517, 5 supporting pair(s), city sign fraction 1.000.
- **AND** VEH-C01 ↔ VEH-C03: score 2.365, 3 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C02 ↔ BLT-C13: score 2.343, 4 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C02 ↔ BLT-C08: score 2.338, 6 supporting pair(s), city sign fraction 0.993.
- **AND** SKY-C02 ↔ SKY-C03: score 2.306, 3 supporting pair(s), city sign fraction 1.000.
- **AND** RDP-C01 ↔ RDP-C02: score 2.283, 1 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C01 ↔ BLT-C08: score 2.277, 3 supporting pair(s), city sign fraction 1.000.
- **AND** VEH-C01 ↔ VEH-C04: score 2.274, 2 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C01 ↔ BLT-C05: score 2.202, 5 supporting pair(s), city sign fraction 0.990.
- **AND** BLT-C01 ↔ BLT-C07: score 2.201, 3 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C04 ↔ BLT-C13: score 2.201, 1 supporting pair(s), city sign fraction 1.000.
- **AND** BLT-C05 ↔ BLT-C07: score 2.187, 2 supporting pair(s), city sign fraction 1.000.
- **AND** TRN-C01 ↔ TRN-C02: score 2.177, 1 supporting pair(s), city sign fraction 0.947.
- **AND** VEH-C01 ↔ VEH-C05: score 2.138, 2 supporting pair(s), city sign fraction 0.964.
- **AND** BLT-C01 ↔ BLT-C03: score 2.079, 1 supporting pair(s), city sign fraction 1.000.
- **AND** VEG-C01 ↔ VEG-C08: score 2.073, 1 supporting pair(s), city sign fraction 0.926.
- **AND** VEG-C01 ↔ VEG-C02: score 1.862, 1 supporting pair(s), city sign fraction 1.000.
- **OR** BLT-C03 ↔ BLT-C07: score 1.022, 1 supporting pair(s), city sign fraction 1.000.
- **OR** SKY-C03 ↔ SKY-C06: score 0.807, 1 supporting pair(s), city sign fraction 1.000.

## Files

- Data: `/tmp/uvg-typed-hierarchy-v2/paper/data/feature_mae_semantic_parent_child_v2`
- Figures: `/tmp/uvg-typed-hierarchy-v2/paper/figures/supplementary/feature_mae_semantic_parent_child_v2`
- Machine-readable hierarchy: `typed_visual_hierarchy.json`
