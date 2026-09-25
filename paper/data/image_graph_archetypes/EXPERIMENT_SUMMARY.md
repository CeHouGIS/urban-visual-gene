# Image graph archetype experiment summary

## Required execution sequence

- **500-image graph QA:** passed; node sums, 364 adjacency pairs, fixed edge ordering, finite values, and 20 random visual checks were verified.
- **5,000-image end-to-end smoke test:** passed; graph features, PCA, K=8 clustering, prototypes, representatives, atlas, and city output were generated without error.
- **Full experiment:** passed on all 378,818 images after the two smoke stages.
- All numerical stages ran in isolated subprocesses with logical CPUs 8 and 9 excluded.

## Configuration

- Total images: **378,818**
- Training images: **60,000**
- Number of cities: **30**
- Graph nodes: **64 fixed F categories**
- Possible undirected edges: **2,016**
- Descriptor: **64 node areas + 2,016 normalized 4-neighbour edge contacts = 2,080D**
- PCA components: **44**
- PCA explained variance: **90.0118%**
- Clustering: **MiniBatchKMeans, K=8**
- KMeans parameters: `batch_size=4096, n_init=20, max_iter=300, random_state=42`
- Semantic labels used in computation: **No**
- City used in clustering features: **No**

## Cluster summary

| Archetype | Images | Share | Top nodes | Top edges |
|---|---:|---:|---|---|
| A01 | 56,315 | 14.87% | F029, F013, F058 | F029-F054, F013-F058, F027-F058 |
| A02 | 55,346 | 14.61% | F028, F010, F054 | F010-F013, F036-F054, F033-F054 |
| A03 | 54,987 | 14.52% | F021, F053, F013 | F007-F013, F007-F010, F007-F045 |
| A04 | 52,592 | 13.88% | F054, F058, F013 | F036-F054, F021-F054, F029-F054 |
| A05 | 46,671 | 12.32% | F036, F013, F054 | F036-F054, F007-F013, F029-F036 |
| A06 | 44,879 | 11.85% | F058, F054, F036 | F036-F054, F013-F058, F007-F013 |
| A07 | 37,427 | 9.88% | F057, F054, F058 | F036-F054, F007-F057, F013-F058 |
| A08 | 30,601 | 8.08% | F057, F054, F013 | F036-F054, F010-F013, F007-F010 |

## Representative image sources

Each archive path is paired with the stored JPEG offset and byte size.

### A01

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000064.tar`; offset=697499648; size=88744; city=Turkey/Istanbul; image_id=333525; distance=0.079789
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Japan/Osaka/shard-000659.tar`; offset=256511488; size=72003; city=Japan/Osaka; image_id=158765; distance=0.081539
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Singapore/Singapore/shard-000043.tar`; offset=662849536; size=91963; city=Singapore/Singapore; image_id=261441; distance=0.082244
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Japan/Osaka/shard-000659.tar`; offset=276802048; size=71402; city=Japan/Osaka; image_id=160357; distance=0.082682
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Russia/Moscow/shard-000086.tar`; offset=324402176; size=92770; city=Russia/Moscow; image_id=252178; distance=0.083480
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000442.tar`; offset=389117952; size=103215; city=UnitedKingdom/London; image_id=344278; distance=0.083691
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000414.tar`; offset=556830208; size=66974; city=Indonesia/Jakarta; image_id=142706; distance=0.084252
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000448.tar`; offset=288926208; size=99798; city=Argentina/BuenosAires; image_id=10600; distance=0.084806
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Russia/Moscow/shard-000206.tar`; offset=409173504; size=70947; city=Russia/Moscow; image_id=246585; distance=0.085194
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000040.tar`; offset=830211072; size=109470; city=Indonesia/Jakarta; image_id=140659; distance=0.085444
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000839.tar`; offset=47562752; size=110595; city=Brazil/SaoPaulo; image_id=51978; distance=0.085536
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Austria/Vienna/shard-000073.tar`; offset=54281216; size=81034; city=Austria/Vienna; image_id=34835; distance=0.085798

### A02

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000101.tar`; offset=548939776; size=96536; city=India/Mumbai; image_id=119406; distance=0.075522
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000048.tar`; offset=388917248; size=115153; city=India/Mumbai; image_id=118690; distance=0.081039
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000977.tar`; offset=13920768; size=94921; city=China/HongKong; image_id=83398; distance=0.084611
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000440.tar`; offset=96891392; size=102367; city=Turkey/Istanbul; image_id=332376; distance=0.085496
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Austria/Vienna/shard-000043.tar`; offset=289576448; size=97451; city=Austria/Vienna; image_id=36827; distance=0.085734
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000029.tar`; offset=545340416; size=70305; city=SouthKorea/Seoul; image_id=299685; distance=0.086374
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000064.tar`; offset=526699520; size=73005; city=Turkey/Istanbul; image_id=337337; distance=0.087115
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Austria/Vienna/shard-000084.tar`; offset=146303488; size=89903; city=Austria/Vienna; image_id=35771; distance=0.087380
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000513.tar`; offset=482134016; size=103384; city=Argentina/BuenosAires; image_id=12367; distance=0.087488
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000134.tar`; offset=648850432; size=122160; city=Philippines/Manila; image_id=229577; distance=0.087569
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000233.tar`; offset=67451392; size=77385; city=Turkey/Istanbul; image_id=332415; distance=0.087962
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Austria/Vienna/shard-000045.tar`; offset=43453440; size=86416; city=Austria/Vienna; image_id=29889; distance=0.088131

### A03

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000001.tar`; offset=678469120; size=129291; city=Peru/Lima; image_id=221261; distance=0.086886
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000769.tar`; offset=559652352; size=106273; city=Argentina/BuenosAires; image_id=8040; distance=0.087904
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000442.tar`; offset=382736896; size=98222; city=UnitedKingdom/London; image_id=353397; distance=0.088830
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000191.tar`; offset=321777664; size=114783; city=Malaysia/KualaLumpur; image_id=164589; distance=0.090482
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000409.tar`; offset=322135040; size=83493; city=Philippines/Manila; image_id=234396; distance=0.092250
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000441.tar`; offset=102748672; size=100205; city=UnitedStates/LosAngeles; image_id=356741; distance=0.093101
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000092.tar`; offset=35984896; size=107478; city=India/Mumbai; image_id=123964; distance=0.093191
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Bangladesh/Dhaka/shard-000165.tar`; offset=52475392; size=104604; city=Bangladesh/Dhaka; image_id=49964; distance=0.093759
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000092.tar`; offset=175237120; size=103630; city=India/Mumbai; image_id=117053; distance=0.094362
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Australia/Sydney/shard-000316.tar`; offset=509212160; size=78961; city=Australia/Sydney; image_id=16655; distance=0.094689
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000025.tar`; offset=592201728; size=127367; city=India/Mumbai; image_id=123080; distance=0.094872
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000060.tar`; offset=646096384; size=115360; city=India/Mumbai; image_id=118955; distance=0.094873

### A04

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000095.tar`; offset=853858816; size=81934; city=Philippines/Manila; image_id=238376; distance=0.078587
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000164.tar`; offset=632347648; size=70851; city=Argentina/BuenosAires; image_id=10395; distance=0.080590
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000323.tar`; offset=387151872; size=57795; city=Peru/Lima; image_id=220380; distance=0.083536
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000441.tar`; offset=550887424; size=66937; city=UnitedStates/LosAngeles; image_id=362230; distance=0.083938
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000513.tar`; offset=237714432; size=81967; city=Argentina/BuenosAires; image_id=5743; distance=0.083952
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000220.tar`; offset=70642688; size=59700; city=Peru/Lima; image_id=226315; distance=0.084634
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000854.tar`; offset=555004416; size=67647; city=Argentina/BuenosAires; image_id=5593; distance=0.085290
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000854.tar`; offset=720752128; size=61188; city=Argentina/BuenosAires; image_id=10256; distance=0.085908
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000483.tar`; offset=98355712; size=84702; city=Mexico/MexicoCity; image_id=185141; distance=0.086001
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Mexico/MexicoCity/shard-000483.tar`; offset=377041920; size=74670; city=Mexico/MexicoCity; image_id=183235; distance=0.086266
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000281.tar`; offset=869135872; size=78847; city=Argentina/BuenosAires; image_id=5710; distance=0.086376
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000196.tar`; offset=505765376; size=78112; city=SouthAfrica/Johannesburg; image_id=284714; distance=0.086380

### A05

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000207.tar`; offset=756974080; size=79394; city=UnitedStates/NewYorkCity; image_id=376623; distance=0.084647
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000096.tar`; offset=540489216; size=124842; city=UnitedStates/NewYorkCity; image_id=367756; distance=0.085097
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Bangladesh/Dhaka/shard-000145.tar`; offset=748977664; size=67999; city=Bangladesh/Dhaka; image_id=43821; distance=0.085478
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000138.tar`; offset=793375744; size=80571; city=Malaysia/KualaLumpur; image_id=166332; distance=0.085590
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000006.tar`; offset=669478400; size=56490; city=Taiwan/Taipei; image_id=306578; distance=0.086147
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Russia/Moscow/shard-000232.tar`; offset=614537216; size=96689; city=Russia/Moscow; image_id=250742; distance=0.086642
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000006.tar`; offset=323871744; size=105681; city=Taiwan/Taipei; image_id=304903; distance=0.087505
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000226.tar`; offset=341325824; size=93326; city=UnitedStates/NewYorkCity; image_id=373057; distance=0.087753
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000121.tar`; offset=21914112; size=111424; city=China/HongKong; image_id=85869; distance=0.087906
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000359.tar`; offset=416695296; size=56677; city=UnitedKingdom/London; image_id=342446; distance=0.088178
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Taiwan/Taipei/shard-000175.tar`; offset=464324096; size=101722; city=Taiwan/Taipei; image_id=304244; distance=0.088356
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Russia/Moscow/shard-000216.tar`; offset=512624128; size=86948; city=Russia/Moscow; image_id=243807; distance=0.088427

### A06

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000164.tar`; offset=265015808; size=63369; city=Argentina/BuenosAires; image_id=9138; distance=0.085102
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000313.tar`; offset=568282624; size=75794; city=Peru/Lima; image_id=226419; distance=0.085382
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000633.tar`; offset=45641728; size=74151; city=Argentina/BuenosAires; image_id=7407; distance=0.086977
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000198.tar`; offset=234644992; size=95540; city=Indonesia/Jakarta; image_id=141992; distance=0.088071
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000274.tar`; offset=19523584; size=98527; city=Turkey/Istanbul; image_id=334314; distance=0.088131
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/France/Paris/shard-000025.tar`; offset=786360320; size=82646; city=France/Paris; image_id=105833; distance=0.089162
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000137.tar`; offset=707878912; size=66783; city=Philippines/Manila; image_id=236819; distance=0.089493
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Argentina/BuenosAires/shard-000281.tar`; offset=712553472; size=85695; city=Argentina/BuenosAires; image_id=5927; distance=0.089650
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/Turkey/Istanbul/shard-000094.tar`; offset=229195264; size=67723; city=Turkey/Istanbul; image_id=331004; distance=0.090906
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/China/HongKong/shard-000114.tar`; offset=33624576; size=84084; city=China/HongKong; image_id=77742; distance=0.091997
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/Singapore/Singapore/shard-000087.tar`; offset=719464448; size=119216; city=Singapore/Singapore; image_id=259144; distance=0.092083
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/Thailand/Bangkok/shard-000223.tar`; offset=875679744; size=85637; city=Thailand/Bangkok; image_id=317705; distance=0.092466

### A07

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Thailand/Bangkok/shard-000477.tar`; offset=97361408; size=75237; city=Thailand/Bangkok; image_id=319620; distance=0.071918
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000029.tar`; offset=564301824; size=69650; city=SouthKorea/Seoul; image_id=298359; distance=0.072887
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Colombia/Bogota/shard-000128.tar`; offset=463306240; size=65083; city=Colombia/Bogota; image_id=91014; distance=0.074790
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/Canada/Toronto/shard-000159.tar`; offset=537876992; size=60472; city=Canada/Toronto; image_id=74242; distance=0.074818
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Brazil/SaoPaulo/shard-000839.tar`; offset=439539200; size=76935; city=Brazil/SaoPaulo; image_id=56092; distance=0.075263
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Australia/Sydney/shard-000263.tar`; offset=317655040; size=75221; city=Australia/Sydney; image_id=21394; distance=0.076078
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000166.tar`; offset=318519296; size=63830; city=SouthAfrica/Johannesburg; image_id=285693; distance=0.078083
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Malaysia/KualaLumpur/shard-000164.tar`; offset=854504960; size=66918; city=Malaysia/KualaLumpur; image_id=173177; distance=0.078934
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/SouthKorea/Seoul/shard-000170.tar`; offset=52928512; size=56571; city=SouthKorea/Seoul; image_id=294126; distance=0.079443
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000441.tar`; offset=95540736; size=57965; city=UnitedStates/LosAngeles; image_id=355684; distance=0.079581
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/SouthAfrica/Johannesburg/shard-000087.tar`; offset=457270272; size=76555; city=SouthAfrica/Johannesburg; image_id=278139; distance=0.079750
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/France/Paris/shard-000074.tar`; offset=63242752; size=65148; city=France/Paris; image_id=110767; distance=0.080693

### A08

- Rank 1: `/nas_data_24T/GSV/packages_main/cities/Canada/Toronto/shard-000102.tar`; offset=183513600; size=53723; city=Canada/Toronto; image_id=69206; distance=0.083523
- Rank 2: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000286.tar`; offset=541476352; size=83517; city=UnitedStates/NewYorkCity; image_id=377895; distance=0.083659
- Rank 3: `/nas_data_24T/GSV/packages_main/cities/Bangladesh/Dhaka/shard-000037.tar`; offset=531591680; size=91496; city=Bangladesh/Dhaka; image_id=43457; distance=0.084840
- Rank 4: `/nas_data_24T/GSV/packages_main/cities/UnitedKingdom/London/shard-000359.tar`; offset=426738176; size=66471; city=UnitedKingdom/London; image_id=341183; distance=0.085732
- Rank 5: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000072.tar`; offset=752991232; size=77850; city=Indonesia/Jakarta; image_id=139241; distance=0.086481
- Rank 6: `/nas_data_24T/GSV/packages_main/cities/Philippines/Manila/shard-000338.tar`; offset=64933888; size=88809; city=Philippines/Manila; image_id=238235; distance=0.086707
- Rank 7: `/nas_data_24T/GSV/packages_main/cities/Netherlands/Amsterdam/shard-000042.tar`; offset=751789568; size=81759; city=Netherlands/Amsterdam; image_id=200857; distance=0.086978
- Rank 8: `/nas_data_24T/GSV/packages_main/cities/Peru/Lima/shard-000181.tar`; offset=431113216; size=65843; city=Peru/Lima; image_id=226563; distance=0.087624
- Rank 9: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/LosAngeles/shard-000339.tar`; offset=651023872; size=78486; city=UnitedStates/LosAngeles; image_id=363655; distance=0.087770
- Rank 10: `/nas_data_24T/GSV/packages_main/cities/Indonesia/Jakarta/shard-000040.tar`; offset=788875264; size=83922; city=Indonesia/Jakarta; image_id=147213; distance=0.087775
- Rank 11: `/nas_data_24T/GSV/packages_main/cities/UnitedStates/NewYorkCity/shard-000160.tar`; offset=772201472; size=104444; city=UnitedStates/NewYorkCity; image_id=371270; distance=0.087983
- Rank 12: `/nas_data_24T/GSV/packages_main/cities/India/Mumbai/shard-000101.tar`; offset=509942784; size=66635; city=India/Mumbai; image_id=120834; distance=0.088563

## Generated figures

- `paper/figures/main/Fig_Graph_QA.png`
- `paper/figures/main/Fig_Graph_Composition_Archetype_Atlas.png`
- `paper/figures/main/Fig_Graph_Composition_Archetype_Atlas.pdf`
- `paper/figures/main/Fig_City_Archetype_Heatmap.png`
- `paper/figures/main/Fig_City_Archetype_Heatmap.pdf`

## Interpretation boundary

These are recurring image-level visual-composition graph prototypes. They are not city-level clusters, and semantic annotations were not used for graph construction, PCA, or clustering.
