# Compact spectral image-graph archetype experiment

## Encoding

Each image is encoded in **201 dimensions**, reduced from the original 2,080D descriptor:

- 64 square-root node-area coordinates;
- 1 cross-category boundary-density coordinate;
- 136 upper-triangle coordinates from a rank-16 spectral projection of the composition-corrected residual adjacency matrix.

The residual compares observed cross-category contacts against independent mixing conditional on the image's node composition. A fixed normalized-Laplacian basis learned from the city-balanced training set makes all image encodings directly comparable. Semantic labels and city labels were not clustering inputs.

## Execution and QA

- Total images: **378,818**
- Training images: **60,000**
- Cities: **30**
- Feature shape: **378,818 × 201**, float32
- PCA: **112 components**, **90.0376%** explained variance
- Clustering: **MiniBatchKMeans, K=8, random seed=42**
- Node PCA loading energy: **38.74%**
- Boundary-density PCA loading energy: **0.14%**
- Spectral-topology PCA loading energy: **61.12%**
- Maximum node normalization error: `1.192e-07`
- Random 20-image exact rebuild error: `0.000e+00`
- NaN/Inf: **0**

## Descriptive cluster diagnostic

- Mean silhouette on a fixed 5,000-image sample: **0.0167**
- Negative-silhouette share: **22.18%**
- This diagnostic is descriptive, not a K-selection step. The compact representation improves topology participation, but it does not make the naturally continuous street-scene distribution sharply separated.

## Cluster summary

| Archetype | Images | Share | Top nodes | Top edges |
|---|---:|---:|---|---|
| A01 | 63,130 | 16.66% | F057, F054, F058 | F036-F054, F007-F013, F007-F057 |
| A02 | 54,961 | 14.51% | F058, F054, F057 | F036-F054, F013-F058, F001-F013 |
| A03 | 49,794 | 13.14% | F058, F029, F013 | F013-F058, F036-F054, F027-F058 |
| A04 | 46,644 | 12.31% | F036, F013, F007 | F036-F054, F007-F013, F001-F013 |
| A05 | 46,445 | 12.26% | F028, F054, F021 | F010-F013, F033-F054, F021-F028 |
| A06 | 42,898 | 11.32% | F054, F036, F013 | F036-F054, F007-F013, F001-F013 |
| A07 | 39,879 | 10.53% | F054, F028, F058 | F036-F054, F010-F013, F021-F054 |
| A08 | 35,067 | 9.26% | F054, F010, F021 | F010-F013, F036-F054, F027-F035 |

## Representative images

### A01

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000199.tar`; offset=510162432; size=91160; city=UnitedStates/NewYorkCity; image_id=368570; distance=0.607584
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000094.tar`; offset=433880064; size=51661; city=Turkey/Istanbul; image_id=340658; distance=0.620645
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000128.tar`; offset=485251072; size=62675; city=Colombia/Bogota; image_id=98956; distance=0.625549
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000813.tar`; offset=342076928; size=78356; city=Brazil/SaoPaulo; image_id=51756; distance=0.636553
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000220.tar`; offset=183001088; size=76053; city=Peru/Lima; image_id=217105; distance=0.638573
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000124.tar`; offset=501406720; size=81335; city=Philippines/Manila; image_id=231200; distance=0.640749
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000522.tar`; offset=693504000; size=72668; city=Mexico/MexicoCity; image_id=186074; distance=0.641936
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/CapeTown/shard-000159.tar`; offset=499721216; size=81696; city=SouthAfrica/CapeTown; image_id=270218; distance=0.643587
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000483.tar`; offset=466442240; size=55507; city=Mexico/MexicoCity; image_id=179991; distance=0.650980
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Canada/Toronto/shard-000055.tar`; offset=150627328; size=76972; city=Canada/Toronto; image_id=69937; distance=0.652937
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000246.tar`; offset=151597056; size=84650; city=Brazil/SaoPaulo; image_id=51902; distance=0.653116
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000254.tar`; offset=814917120; size=90664; city=UnitedKingdom/London; image_id=353259; distance=0.654608

### A02

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000496.tar`; offset=37248000; size=60550; city=Indonesia/Jakarta; image_id=147311; distance=0.603574
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000006.tar`; offset=8385024; size=68719; city=Taiwan/Taipei; image_id=313815; distance=0.605124
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000127.tar`; offset=183798272; size=63965; city=UnitedStates/LosAngeles; image_id=355791; distance=0.623390
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000201.tar`; offset=830828032; size=63816; city=Indonesia/Jakarta; image_id=142982; distance=0.629415
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000164.tar`; offset=287752704; size=68932; city=Malaysia/KualaLumpur; image_id=175789; distance=0.629511
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000048.tar`; offset=648355840; size=46737; city=India/Mumbai; image_id=115489; distance=0.629784
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Bangladesh/Dhaka/shard-000116.tar`; offset=590992384; size=55327; city=Bangladesh/Dhaka; image_id=46620; distance=0.630138
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000166.tar`; offset=161134080; size=50189; city=SouthAfrica/Johannesburg; image_id=290278; distance=0.631943
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000196.tar`; offset=119205376; size=62109; city=SouthAfrica/Johannesburg; image_id=287755; distance=0.632685
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000179.tar`; offset=198862848; size=64306; city=Colombia/Bogota; image_id=93790; distance=0.634119
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000465.tar`; offset=599199744; size=73111; city=SouthAfrica/Johannesburg; image_id=283994; distance=0.634851
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000013.tar`; offset=161403904; size=61111; city=Malaysia/KualaLumpur; image_id=172251; distance=0.635862

### A03

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Singapore/Singapore/shard-000090.tar`; offset=686771712; size=100656; city=Singapore/Singapore; image_id=255578; distance=0.575299
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000029.tar`; offset=304986624; size=61617; city=SouthKorea/Seoul; image_id=302656; distance=0.617648
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Australia/Sydney/shard-000315.tar`; offset=269337600; size=106242; city=Australia/Sydney; image_id=18136; distance=0.628933
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000145.tar`; offset=495110656; size=83428; city=UnitedStates/LosAngeles; image_id=363895; distance=0.633730
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000364.tar`; offset=254728704; size=64130; city=Malaysia/KualaLumpur; image_id=173987; distance=0.635216
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Australia/Sydney/shard-000211.tar`; offset=160371712; size=89100; city=Australia/Sydney; image_id=24659; distance=0.636711
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000220.tar`; offset=651522048; size=57546; city=Peru/Lima; image_id=214216; distance=0.638387
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Canada/Toronto/shard-000025.tar`; offset=73297920; size=66577; city=Canada/Toronto; image_id=75960; distance=0.638926
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000769.tar`; offset=19660800; size=63726; city=Argentina/BuenosAires; image_id=5933; distance=0.639314
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Canada/Toronto/shard-000016.tar`; offset=104868352; size=75355; city=Canada/Toronto; image_id=66942; distance=0.641374
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000273.tar`; offset=6609408; size=113488; city=China/HongKong; image_id=78490; distance=0.648576
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000160.tar`; offset=840310272; size=112868; city=UnitedStates/NewYorkCity; image_id=372732; distance=0.650470

### A04

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Japan/Osaka/shard-000040.tar`; offset=391941632; size=94883; city=Japan/Osaka; image_id=162285; distance=0.626427
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000101.tar`; offset=447047168; size=120992; city=India/Mumbai; image_id=123960; distance=0.652212
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000147.tar`; offset=673232384; size=74936; city=SouthKorea/Seoul; image_id=292351; distance=0.670420
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000013.tar`; offset=543664640; size=90907; city=Malaysia/KualaLumpur; image_id=171554; distance=0.674695
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Japan/Osaka/shard-000659.tar`; offset=535558656; size=74291; city=Japan/Osaka; image_id=157930; distance=0.675349
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000087.tar`; offset=156358144; size=81304; city=SouthAfrica/Johannesburg; image_id=281958; distance=0.678896
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Russia/Moscow/shard-000216.tar`; offset=166767104; size=142419; city=Russia/Moscow; image_id=252013; distance=0.679684
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000364.tar`; offset=34576896; size=87475; city=Malaysia/KualaLumpur; image_id=171089; distance=0.688445
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000226.tar`; offset=419961856; size=97770; city=UnitedStates/NewYorkCity; image_id=375549; distance=0.692418
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000274.tar`; offset=350740992; size=76033; city=Turkey/Istanbul; image_id=339656; distance=0.695940
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000128.tar`; offset=20017152; size=79926; city=Colombia/Bogota; image_id=98153; distance=0.696563
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000254.tar`; offset=592890880; size=66494; city=UnitedKingdom/London; image_id=343842; distance=0.696653

### A05

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000401.tar`; offset=390896128; size=94461; city=Philippines/Manila; image_id=233498; distance=0.628026
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000286.tar`; offset=389465600; size=76532; city=Philippines/Manila; image_id=236853; distance=0.631467
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000072.tar`; offset=264019456; size=67471; city=Indonesia/Jakarta; image_id=139584; distance=0.637193
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000414.tar`; offset=248634880; size=111291; city=Indonesia/Jakarta; image_id=140501; distance=0.654340
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000023.tar`; offset=756262912; size=69232; city=Colombia/Bogota; image_id=95978; distance=0.654976
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000074.tar`; offset=597817856; size=79829; city=Taiwan/Taipei; image_id=303383; distance=0.661560
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/India/NewDelhi/shard-000321.tar`; offset=257618944; size=82975; city=India/NewDelhi; image_id=127891; distance=0.662290
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000207.tar`; offset=280988672; size=101538; city=Taiwan/Taipei; image_id=313414; distance=0.662816
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Japan/Osaka/shard-000718.tar`; offset=794685952; size=86285; city=Japan/Osaka; image_id=156692; distance=0.662904
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000029.tar`; offset=338768384; size=72550; city=SouthKorea/Seoul; image_id=290402; distance=0.665441
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000092.tar`; offset=644759040; size=139401; city=India/Mumbai; image_id=118706; distance=0.666619
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000100.tar`; offset=130392576; size=79947; city=Turkey/Istanbul; image_id=337486; distance=0.667274

### A06

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000287.tar`; offset=243494912; size=123203; city=UnitedStates/NewYorkCity; image_id=375393; distance=0.651696
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Thailand/Bangkok/shard-000576.tar`; offset=155791872; size=115914; city=Thailand/Bangkok; image_id=323774; distance=0.663108
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000266.tar`; offset=6891520; size=71036; city=China/HongKong; image_id=83504; distance=0.665469
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000522.tar`; offset=745746944; size=97165; city=Mexico/MexicoCity; image_id=186535; distance=0.667205
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000332.tar`; offset=103839744; size=85751; city=SouthKorea/Seoul; image_id=294154; distance=0.668777
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/CapeTown/shard-000029.tar`; offset=570658304; size=66047; city=SouthAfrica/CapeTown; image_id=267675; distance=0.669592
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Singapore/Singapore/shard-000034.tar`; offset=389081600; size=120157; city=Singapore/Singapore; image_id=261754; distance=0.671720
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Australia/Sydney/shard-000263.tar`; offset=775757824; size=117899; city=Australia/Sydney; image_id=18434; distance=0.671827
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Thailand/Bangkok/shard-000223.tar`; offset=449771520; size=110736; city=Thailand/Bangkok; image_id=327015; distance=0.673281
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000092.tar`; offset=263166464; size=65468; city=India/Mumbai; image_id=124628; distance=0.673629
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000313.tar`; offset=360915968; size=83877; city=Peru/Lima; image_id=220201; distance=0.675178
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Bangladesh/Dhaka/shard-000118.tar`; offset=191501312; size=129590; city=Bangladesh/Dhaka; image_id=46091; distance=0.675269

### A07

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000256.tar`; offset=472771072; size=88155; city=Mexico/MexicoCity; image_id=187878; distance=0.641248
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000002.tar`; offset=515419648; size=77783; city=Mexico/MexicoCity; image_id=185097; distance=0.641748
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000181.tar`; offset=333776384; size=97891; city=Peru/Lima; image_id=222137; distance=0.642587
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000633.tar`; offset=567745536; size=84879; city=Argentina/BuenosAires; image_id=3582; distance=0.643451
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000441.tar`; offset=263825920; size=56329; city=UnitedStates/LosAngeles; image_id=360773; distance=0.648107
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000175.tar`; offset=634128384; size=74458; city=Taiwan/Taipei; image_id=303214; distance=0.650374
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000483.tar`; offset=521554944; size=83002; city=Mexico/MexicoCity; image_id=183020; distance=0.650571
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000943.tar`; offset=32507392; size=88270; city=China/HongKong; image_id=81966; distance=0.654524
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000842.tar`; offset=8566272; size=85014; city=China/HongKong; image_id=82840; distance=0.655108
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000110.tar`; offset=296716800; size=75321; city=Brazil/SaoPaulo; image_id=61163; distance=0.655632
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/CapeTown/shard-000163.tar`; offset=40556544; size=72662; city=SouthAfrica/CapeTown; image_id=275113; distance=0.658699
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000002.tar`; offset=801140736; size=81850; city=Mexico/MexicoCity; image_id=187937; distance=0.663909

### A08

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000075.tar`; offset=454499328; size=97425; city=Colombia/Bogota; image_id=101654; distance=0.634343
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/India/NewDelhi/shard-000235.tar`; offset=261139968; size=66776; city=India/NewDelhi; image_id=135100; distance=0.636828
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000001.tar`; offset=408236032; size=81979; city=Peru/Lima; image_id=218107; distance=0.643393
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000137.tar`; offset=537124864; size=67545; city=Colombia/Bogota; image_id=100238; distance=0.648980
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000338.tar`; offset=271681536; size=68102; city=Philippines/Manila; image_id=238894; distance=0.649032
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000170.tar`; offset=768098304; size=94233; city=SouthKorea/Seoul; image_id=294180; distance=0.665185
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000075.tar`; offset=52958720; size=81750; city=UnitedKingdom/London; image_id=352167; distance=0.665641
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/France/Paris/shard-000040.tar`; offset=771431424; size=82546; city=France/Paris; image_id=113309; distance=0.666015
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000075.tar`; offset=29846528; size=73592; city=Colombia/Bogota; image_id=97540; distance=0.669079
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000181.tar`; offset=694112768; size=67489; city=Peru/Lima; image_id=218865; distance=0.675043
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000110.tar`; offset=472983040; size=106607; city=Brazil/SaoPaulo; image_id=56206; distance=0.675066
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000156.tar`; offset=612030464; size=70621; city=Malaysia/KualaLumpur; image_id=165813; distance=0.675093

## Generated figures

- `paper/figures/main/Fig_Compact_Graph_QA.png`
- `paper/figures/main/Fig_Compact_Graph_Archetype_Atlas.png`
- `paper/figures/main/Fig_Compact_Graph_Archetype_Atlas.pdf`
- `paper/figures/main/Fig_Compact_Graph_City_Heatmap.png`
- `paper/figures/main/Fig_Compact_Graph_City_Heatmap.pdf`

## Interpretation boundary

The clusters describe recurring layouts within single directional street-view images. They are not city-level clusters. Original 2,080D node/edge means are used only after clustering to visualize each prototype.
