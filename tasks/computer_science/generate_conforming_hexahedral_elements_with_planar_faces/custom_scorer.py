"""Custom scorers for the octree hanging-node-removal (all-hex template) task.

Three weighted scorers mirror the author's reference checker ``check.py``:

  mesh_quality      (weight 10) - every cell is a VTK_HEXAHEDRON whose minimum
                                  scaled Jacobian over its eight corners is > 0.21
  coplanarity       (weight 10) - the four vertices of each of the six faces of
                                  every hex are coplanar within 1e-6
  mesh_consistency  (weight 80) - element-count budget, no ghost vertices,
                                  conforming face sharing, boundary faces on the
                                  cube surface, and full coverage of the private
                                  reference octree point set (no original or
                                  hanging node may be deleted)

Contract notes
--------------
* The agent must write ``0.vtk`` ... ``2047.vtk`` to the workspace root.  If the
  root does not hold them, the single sub-directory containing the most required
  files is scored instead, and the directory actually used is reported in
  ``details`` (agents that run the program inside ``data/`` are not silently
  zeroed).
* Unlike ``check.py``, all 2048 files are *required*: a missing file fails every
  flag for that state, so a partial submission cannot collect full credit.
* The reference octree point sets are read from ``ref_dir/octree/<i>_ref.vtk``.
  Scorers never regenerate ground truth; they only read the materialised
  reference directory, as required by the framework's runtime boundary.
* Analysis is fully vectorised (numpy) and cached per output directory, so the
  three scorers share one pass: ~20 s for all 2048 meshes on 2 cores.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

N_STATE_PAIRS = 2048

# Per-state hex-element budget (public: identical table appears in the prompts).
THRES = [
    536, 608, 576, 648, 608, 616, 648, 672, 608, 756, 662, 816, 694, 784, 754, 852,
    584, 718, 686, 826, 670, 746, 778, 862, 654, 746, 734, 832, 732, 808, 818, 868,
    584, 670, 686, 778, 718, 746, 840, 876, 656, 810, 736, 896, 796, 836, 896, 966,
    616, 770, 704, 822, 770, 782, 836, 864, 734, 832, 786, 856, 838, 872, 904, 954,
    608, 732, 686, 830, 742, 822, 854, 934, 616, 814, 728, 904, 770, 840, 874, 952,
    656, 834, 760, 958, 796, 874, 934, 1034, 738, 838, 806, 948, 814, 892, 966, 1052,
    654, 770, 772, 908, 732, 846, 876, 956, 738, 858, 820, 1010, 794, 892, 924, 1058,
    734, 876, 824, 980, 818, 910, 900, 1028, 664, 780, 742, 872, 736, 802, 834, 908,
    576, 700, 658, 774, 662, 742, 736, 804, 648, 860, 736, 912, 754, 880, 834, 948,
    648, 854, 756, 954, 740, 874, 854, 990, 780, 886, 824, 958, 836, 934, 908, 994,
    624, 754, 732, 868, 778, 836, 892, 966, 696, 920, 802, 992, 862, 932, 948, 1062,
    680, 898, 774, 950, 846, 910, 912, 992, 824, 930, 876, 1010, 928, 998, 994, 1080,
    648, 822, 760, 940, 802, 904, 906, 1016, 672, 926, 790, 1016, 838, 952, 942, 1064,
    720, 968, 850, 1078, 886, 994, 1010, 1154, 844, 980, 912, 1090, 920, 1034, 1072, 1194,
    742, 866, 824, 990, 798, 928, 928, 1038, 806, 970, 888, 1122, 862, 1004, 992, 1170,
    810, 996, 900, 1100, 866, 1030, 1004, 1148, 758, 948, 902, 1074, 868, 1004, 1000, 1116,
    584, 654, 624, 742, 656, 738, 696, 806, 718, 746, 778, 812, 796, 808, 862, 876,
    694, 708, 802, 822, 772, 798, 886, 858, 708, 816, 794, 908, 786, 884, 878, 944,
    632, 708, 754, 822, 772, 790, 900, 920, 772, 800, 858, 892, 862, 888, 976, 990,
    746, 794, 798, 846, 822, 834, 888, 916, 794, 908, 846, 932, 898, 948, 964, 1030,
    670, 770, 740, 874, 796, 866, 914, 978, 746, 838, 822, 928, 822, 892, 926, 1004,
    772, 824, 882, 954, 862, 926, 1014, 1058, 798, 914, 866, 1024, 874, 968, 1026, 1128,
    708, 808, 832, 952, 786, 890, 936, 1000, 790, 882, 928, 1006, 874, 944, 1032, 1054,
    814, 900, 904, 976, 898, 934, 1008, 1052, 712, 882, 790, 1046, 784, 1004, 882, 1122,
    686, 786, 732, 838, 736, 834, 802, 902, 840, 882, 892, 934, 896, 930, 948, 998,
    854, 890, 954, 990, 910, 938, 1010, 1054, 880, 1002, 952, 1102, 936, 1050, 1036, 1138,
    754, 846, 854, 946, 858, 942, 972, 1072, 900, 942, 972, 1042, 976, 1038, 1090, 1140,
    898, 934, 950, 986, 974, 1030, 1040, 1112, 924, 1046, 976, 1098, 1028, 1142, 1094, 1224,
    778, 900, 854, 990, 882, 1010, 986, 1122, 876, 948, 952, 1038, 952, 1058, 1056, 1170,
    938, 1004, 1034, 1142, 1014, 1114, 1166, 1246, 916, 1068, 1040, 1206, 1020, 1178, 1172, 1338,
    822, 944, 932, 1096, 878, 1006, 1036, 1144, 920, 992, 1058, 1144, 976, 1054, 1134, 1220,
    966, 1032, 1056, 1136, 1022, 1094, 1132, 1212, 912, 1096, 996, 1200, 962, 1158, 1094, 1276,
    584, 656, 650, 742, 656, 664, 742, 806, 656, 824, 700, 860, 748, 844, 798, 896,
    680, 834, 704, 822, 786, 854, 802, 858, 742, 840, 786, 856, 826, 860, 842, 920,
    680, 786, 704, 802, 834, 854, 836, 872, 744, 904, 754, 920, 890, 944, 920, 962,
    664, 830, 774, 898, 830, 842, 912, 940, 794, 892, 862, 932, 898, 932, 980, 1030,
    656, 786, 724, 874, 810, 882, 898, 978, 664, 874, 766, 948, 830, 900, 918, 996,
    744, 928, 778, 982, 890, 982, 958, 1030, 790, 918, 858, 972, 922, 1000, 962, 1048,
    742, 864, 824, 932, 826, 898, 900, 1008, 790, 966, 872, 1006, 874, 1000, 948, 1054,
    794, 936, 900, 1056, 878, 970, 976, 1104, 712, 828, 872, 1046, 784, 850, 1008, 1122,
    650, 738, 674, 762, 700, 780, 724, 820, 742, 904, 724, 908, 798, 924, 802, 976,
    776, 960, 766, 916, 860, 980, 816, 984, 830, 964, 814, 942, 914, 984, 892, 1044,
    738, 860, 728, 816, 870, 928, 840, 946, 794, 984, 778, 962, 926, 1024, 918, 1042,
    828, 956, 836, 1024, 904, 968, 986, 1066, 882, 988, 950, 1084, 986, 1056, 1068, 1154,
    742, 866, 748, 908, 846, 948, 902, 1044, 806, 970, 806, 1044, 882, 996, 970, 1092,
    832, 1046, 840, 1062, 964, 1100, 994, 1148, 894, 1058, 962, 1112, 1026, 1140, 1066, 1188,
    778, 930, 800, 960, 862, 964, 898, 1074, 842, 1062, 924, 1102, 926, 1096, 1000, 1150,
    868, 1054, 974, 1174, 924, 1088, 1078, 1222, 870, 1000, 956, 1100, 920, 1056, 1026, 1170,
    680, 742, 738, 778, 744, 790, 794, 842, 834, 840, 870, 876, 890, 888, 926, 940,
    872, 864, 860, 852, 928, 912, 916, 916, 864, 978, 880, 966, 948, 998, 936, 1030,
    810, 850, 798, 838, 914, 946, 916, 964, 914, 948, 930, 936, 1018, 1016, 1020, 1062,
    868, 860, 936, 928, 944, 956, 1026, 1054, 888, 974, 956, 1042, 992, 1070, 1074, 1168,
    786, 864, 846, 938, 890, 974, 978, 1070, 854, 890, 914, 964, 930, 1000, 1018, 1096,
    928, 986, 968, 1012, 1032, 1068, 1072, 1144, 912, 1028, 924, 1054, 1016, 1138, 1056, 1186,
    850, 956, 904, 996, 934, 990, 980, 1072, 946, 982, 972, 1022, 1002, 1044, 1048, 1098,
    936, 966, 1042, 1086, 992, 1028, 1118, 1162, 822, 1008, 998, 1128, 894, 1070, 1074, 1204,
    704, 838, 728, 814, 754, 886, 778, 938, 836, 906, 840, 904, 920, 954, 918, 1006,
    904, 968, 860, 918, 988, 988, 938, 1048, 902, 1052, 880, 1024, 986, 1072, 958, 1132,
    798, 918, 782, 868, 930, 986, 894, 1004, 916, 986, 894, 958, 1020, 1054, 1006, 1100,
    900, 992, 968, 1060, 976, 1032, 1058, 1130, 954, 1076, 1022, 1172, 1058, 1144, 1140, 1242,
    802, 924, 802, 960, 906, 1006, 956, 1102, 872, 1000, 932, 1074, 948, 1054, 1036, 1150,
    960, 1054, 962, 1064, 1064, 1136, 1088, 1212, 966, 1146, 1034, 1200, 1098, 1228, 1138, 1276,
    838, 988, 854, 1012, 922, 1022, 952, 1132, 964, 1064, 990, 1132, 1048, 1098, 1094, 1208,
    968, 1062, 1074, 1210, 1052, 1124, 1178, 1258, 936, 1154, 1022, 1182, 986, 1216, 1092, 1350,
    584, 656, 648, 720, 654, 738, 780, 844, 670, 810, 740, 900, 732, 828, 836, 934,
    656, 796, 816, 948, 732, 814, 898, 982, 732, 824, 856, 954, 794, 876, 938, 988,
    694, 772, 854, 938, 708, 798, 880, 916, 772, 876, 910, 1028, 786, 888, 936, 1034,
    784, 860, 880, 956, 832, 872, 934, 990, 852, 950, 948, 1046, 900, 934, 974, 1052,
    718, 834, 840, 976, 732, 846, 880, 988, 746, 866, 860, 994, 794, 892, 928, 1034,
    796, 924, 958, 1114, 810, 950, 998, 1126, 814, 942, 990, 1132, 862, 968, 1030, 1116,
    708, 824, 876, 1012, 802, 922, 996, 1076, 798, 918, 924, 1114, 870, 968, 1044, 1178,
    832, 974, 958, 1114, 932, 1024, 1050, 1178, 750, 866, 864, 994, 876, 1042, 1082, 1196,
    686, 774, 756, 864, 734, 820, 824, 926, 778, 940, 854, 1016, 818, 972, 908, 1078,
    816, 972, 954, 1110, 856, 1004, 1008, 1172, 898, 1004, 1008, 1142, 938, 1036, 1090, 1176,
    802, 896, 954, 1048, 794, 880, 952, 1054, 886, 1020, 1010, 1172, 878, 1032, 1036, 1178,
    922, 1050, 1018, 1146, 914, 1062, 1016, 1180, 1004, 1110, 1100, 1206, 996, 1094, 1098, 1212,
    826, 950, 940, 1078, 818, 948, 952, 1090, 862, 1026, 976, 1154, 854, 1052, 988, 1194,
    948, 1106, 1096, 1296, 940, 1132, 1136, 1308, 982, 1118, 1158, 1308, 974, 1116, 1170, 1320,
    822, 946, 976, 1142, 894, 1024, 1096, 1206, 858, 1050, 1040, 1246, 930, 1128, 1132, 1338,
    942, 1128, 1068, 1268, 1014, 1206, 1160, 1360, 944, 1074, 1064, 1236, 1048, 1190, 1194, 1344,
    616, 734, 680, 810, 734, 664, 824, 758, 770, 832, 846, 880, 838, 750, 928, 882,
    784, 832, 922, 942, 852, 750, 1004, 944, 832, 946, 914, 1028, 900, 890, 996, 1062,
    746, 814, 898, 966, 794, 712, 924, 912, 822, 912, 974, 1036, 898, 798, 1028, 976,
    834, 910, 930, 1006, 910, 796, 1012, 974, 910, 1024, 1006, 1120, 958, 1008, 1032, 1126,
    770, 876, 884, 1004, 818, 788, 924, 956, 782, 902, 896, 1030, 858, 802, 992, 1004,
    860, 974, 1036, 1136, 936, 874, 1104, 1082, 872, 1016, 1048, 1206, 920, 1042, 1088, 1190,
    794, 900, 920, 1040, 894, 890, 1040, 1104, 834, 926, 1016, 1094, 934, 1004, 1136, 1158,
    910, 996, 1036, 1108, 1010, 1046, 1156, 1200, 796, 1038, 910, 1178, 994, 950, 1140, 1092,
    704, 838, 774, 914, 786, 756, 876, 916, 836, 906, 912, 1010, 904, 840, 994, 1006,
    880, 972, 1018, 1082, 948, 878, 1100, 1078, 934, 1056, 1016, 1166, 974, 1088, 1098, 1200,
    798, 918, 950, 1070, 846, 804, 976, 1010, 888, 1014, 1040, 1138, 964, 888, 1094, 1100,
    930, 1050, 1026, 1146, 1006, 924, 1108, 1108, 1012, 1162, 1108, 1258, 1032, 1146, 1162, 1264,
    822, 972, 936, 1100, 842, 872, 1004, 1074, 864, 1020, 978, 1148, 940, 908, 1074, 1116,
    956, 1106, 1132, 1268, 1032, 994, 1200, 1236, 990, 1170, 1166, 1360, 1038, 1196, 1206, 1344,
    846, 968, 972, 1136, 918, 1046, 1092, 1200, 916, 1044, 1098, 1212, 1016, 1122, 1218, 1276,
    1006, 1100, 1132, 1268, 1106, 1178, 1252, 1332, 974, 1192, 1094, 1332, 1112, 1092, 1258, 1332,
    680, 744, 776, 832, 742, 790, 830, 894, 786, 904, 860, 978, 826, 936, 914, 1040,
    862, 966, 880, 984, 902, 998, 934, 1046, 902, 1000, 948, 1018, 970, 1004, 1002, 1080,
    872, 928, 904, 960, 864, 912, 902, 966, 928, 1046, 988, 1078, 948, 1030, 986, 1112,
    912, 988, 1024, 1100, 904, 1000, 1022, 1134, 980, 1050, 1092, 1162, 972, 1034, 1090, 1168,
    834, 928, 946, 1054, 826, 926, 958, 1066, 854, 974, 966, 1100, 846, 1000, 978, 1140,
    966, 1108, 1050, 1178, 986, 1106, 1062, 1218, 998, 1098, 1054, 1168, 990, 1096, 1066, 1180,
    864, 986, 954, 1062, 964, 1036, 1046, 1154, 912, 1060, 974, 1136, 984, 1138, 1066, 1228,
    932, 1074, 1074, 1230, 1004, 1152, 1166, 1322, 866, 982, 1078, 1192, 1008, 1114, 1170, 1284,
    704, 792, 766, 854, 786, 872, 814, 976, 802, 964, 816, 1000, 842, 968, 892, 1072,
    880, 1064, 894, 1044, 948, 1068, 942, 1116, 934, 1068, 942, 1070, 1002, 1072, 1018, 1176,
    860, 982, 860, 976, 880, 938, 880, 1048, 916, 1078, 938, 1094, 936, 1062, 958, 1144,
    930, 1058, 1042, 1170, 978, 1070, 1096, 1204, 1012, 1146, 1124, 1258, 1032, 1130, 1178, 1264,
    822, 974, 902, 1062, 842, 972, 936, 1112, 858, 1022, 970, 1148, 906, 1048, 1038, 1188,
    984, 1170, 1030, 1224, 1004, 1168, 1064, 1280, 1046, 1210, 1102, 1280, 1066, 1180, 1170, 1320,
    852, 1004, 904, 1064, 952, 1054, 1018, 1200, 916, 1136, 1034, 1212, 1016, 1186, 1126, 1276,
    978, 1164, 1120, 1320, 1050, 1214, 1240, 1384, 974, 1104, 1096, 1240, 1112, 1254, 1182, 1424,
    664, 794, 828, 868, 794, 712, 882, 870, 830, 892, 904, 938, 898, 798, 986, 934,
    912, 932, 930, 978, 980, 866, 1012, 974, 904, 1018, 978, 1064, 972, 1022, 1032, 1126,
    868, 936, 900, 968, 888, 822, 954, 936, 944, 1006, 976, 1066, 992, 908, 1058, 1000,
    906, 982, 1018, 1094, 982, 856, 1100, 1056, 982, 1096, 1094, 1208, 1030, 1080, 1120, 1214,
    830, 936, 942, 1062, 878, 836, 982, 1008, 842, 962, 954, 1088, 918, 850, 1050, 1056,
    988, 1074, 1044, 1172, 1036, 990, 1140, 1112, 1000, 1144, 1056, 1214, 1020, 1114, 1124, 1254,
    860, 966, 978, 1070, 960, 1016, 1070, 1162, 956, 1020, 1018, 1124, 1056, 1070, 1138, 1216,
    982, 1068, 1124, 1196, 1082, 1118, 1244, 1288, 856, 1110, 1136, 1266, 1066, 1010, 1228, 1266,
    774, 914, 836, 988, 862, 886, 950, 970, 912, 982, 986, 1084, 980, 1014, 1068, 1032,
    1024, 1088, 1042, 1134, 1092, 1092, 1124, 1110, 1022, 1172, 1096, 1246, 1090, 1176, 1178, 1188,
    936, 1056, 968, 1088, 956, 1012, 1022, 1036, 1026, 1124, 1058, 1184, 1074, 1080, 1140, 1098,
    1018, 1138, 1130, 1250, 1094, 1150, 1212, 1192, 1100, 1250, 1212, 1362, 1120, 1234, 1266, 1368,
    898, 1048, 1010, 1174, 918, 1046, 1078, 1100, 940, 1096, 1052, 1222, 1016, 1122, 1148, 1170,
    1100, 1222, 1156, 1320, 1148, 1192, 1252, 1240, 1134, 1314, 1190, 1384, 1154, 1284, 1258, 1424,
    928, 1078, 1046, 1210, 1028, 1128, 1166, 1182, 1054, 1154, 1116, 1258, 1154, 1204, 1236, 1350,
    1094, 1188, 1236, 1372, 1194, 1266, 1356, 1436, 1056, 1280, 1178, 1436, 1200, 1266, 1362, 1416,
]

TOL = 1e-6                 # coplanarity / coordinate tolerance
SJ_MIN = 0.21              # scaled-Jacobian floor
CUBE_MIN = 0.0
CUBE_MAX = 537600.0        # 1024 * 525
MATCH_TOL = 1e-6           # node-position matching tolerance

# corner -> three neighbours used for the scaled Jacobian (VTK hex ordering)
NEIGHBORS = np.array(
    [(1, 3, 4), (0, 5, 2), (1, 6, 3), (0, 2, 7),
     (0, 7, 5), (1, 4, 6), (2, 5, 7), (3, 6, 4)], dtype=np.int64)
# the six faces of a VTK hexahedron
FACES = np.array(
    [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
     (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], dtype=np.int64)

FLAG_KEYS = ("sj_ok", "coplanar_ok", "count_ok", "ghost_ok",
             "share_ok", "boundary_ok", "octree_ok")
QUALITY_FLAGS = ("sj_ok",)
COPLANARITY_FLAGS = ("coplanar_ok",)
CONSISTENCY_FLAGS = ("count_ok", "ghost_ok", "share_ok", "boundary_ok", "octree_ok")

_MAX_REPORTED = 12         # cap on per-file diagnostics kept in ScoreDetail

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str, tuple[int, int, float]], dict[str, Any]] = {}
_CACHE_LIMIT = 2


# ─────────────────────────────────────────────────────────────────────────────
# VTK I/O
# ─────────────────────────────────────────────────────────────────────────────

def _vtk_modules():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception as exc:                      # pragma: no cover
        raise RuntimeError(
            "the vtk Python package is required by this task's scorers "
            f"(declare it in runtime.packages): {exc}"
        ) from exc
    return vtk, vtk_to_numpy


def _read_hex_mesh(path: Path):
    """Return (points Nx3 float64, cells Mx8 int64) or (points, reason)."""
    vtk, vtk_to_numpy = _vtk_modules()
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        return None, "unreadable_or_empty"
    pts = np.ascontiguousarray(
        vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)
    n_cells = grid.GetNumberOfCells()
    cell_types = vtk.vtkCellTypes()
    grid.GetDistinctCellTypes(cell_types)
    arr = cell_types.GetCellTypesArray()
    distinct = vtk_to_numpy(arr).tolist() if arr is not None else []
    if distinct != [vtk.VTK_HEXAHEDRON]:
        return grid, "non_hexahedral_cells"      # handled by the slow path
    conn = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int64)
    if conn.size != 8 * n_cells:
        return grid, "ragged_connectivity"
    return pts, conn.reshape(-1, 8)


def _read_points(path: Path):
    """Read only the point set of a reference VTK file."""
    if not path.is_file():
        return None
    vtk, vtk_to_numpy = _vtk_modules()
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        return None
    pts = grid.GetPoints() if hasattr(grid, "GetPoints") else None
    if pts is not None and pts.GetNumberOfPoints() > 0:
        return np.ascontiguousarray(
            vtk_to_numpy(pts.GetData()), dtype=np.float64).reshape(-1, 3)
    return np.array([grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())],
                    dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Per-mesh analysis
# ─────────────────────────────────────────────────────────────────────────────

def _point_coverage_ok(mesh_points: np.ndarray, query_points: np.ndarray) -> bool:
    """True when every reference octree node appears in the submitted mesh.

    Exact integer matching first (all octree coordinates are integers), with a
    tolerance-aware bucketed fallback for the few nodes that need it, matching
    the semantics of the reference checker (MATCH_TOL = 1e-6).
    """
    if query_points.shape[0] == 0:
        return True
    if mesh_points.shape[0] == 0:
        return False

    shift = np.int64(1) << 21
    def keys(a: np.ndarray) -> np.ndarray:
        ai = np.rint(a).astype(np.int64)
        return (ai[:, 0] * shift + ai[:, 1]) * shift + ai[:, 2]

    matched = np.isin(keys(query_points), keys(mesh_points))
    if matched.all():
        return True
    if np.abs(query_points - np.rint(query_points)).max(initial=0.0) > MATCH_TOL:
        return False          # reference node is not integral: fall back below

    # tolerance-aware fallback for the unmatched remainder
    todo = query_points[~matched]
    buckets: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for p in mesh_points:
        buckets.setdefault((int(p[0] // MATCH_TOL), int(p[1] // MATCH_TOL),
                            int(p[2] // MATCH_TOL)), []).append((p[0], p[1], p[2]))
    for q in todo:
        kx, ky, kz = (int(q[0] // MATCH_TOL), int(q[1] // MATCH_TOL),
                      int(q[2] // MATCH_TOL))
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for p in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                        if (abs(p[0] - q[0]) <= MATCH_TOL and
                                abs(p[1] - q[1]) <= MATCH_TOL and
                                abs(p[2] - q[2]) <= MATCH_TOL):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return False
    return True


def _analyze_arrays(pts: np.ndarray, cells: np.ndarray, budget: int,
                    oct_pts: np.ndarray | None) -> dict[str, Any]:
    """Vectorised evaluation of one all-hex mesh."""
    flags: dict[str, Any] = {k: True for k in FLAG_KEYS}
    n_points = pts.shape[0]
    n_cells = cells.shape[0]

    if n_cells > budget:
        flags["count_ok"] = False

    P = pts[cells]                                        # (M, 8, 3)
    idx = np.arange(8)
    a = P[:, NEIGHBORS[:, 0], :] - P[:, idx, :]
    b = P[:, NEIGHBORS[:, 1], :] - P[:, idx, :]
    c = P[:, NEIGHBORS[:, 2], :] - P[:, idx, :]
    det = (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
           - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
           + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0]))
    nrm = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
           * np.linalg.norm(c, axis=-1))
    zero = nrm == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sj = np.where(zero, -np.inf, det / np.where(zero, 1.0, nrm))
    if not bool(np.all(sj.min(axis=1) > SJ_MIN)):
        flags["sj_ok"] = False

    F = P[:, FACES, :]                                    # (M, 6, 4, 3)
    p0, p1, p2, p3 = F[:, :, 0], F[:, :, 1], F[:, :, 2], F[:, :, 3]
    nvec = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(nvec, axis=-1)
    dist = (np.abs(np.einsum("...i,...i->...", p3 - p0, nvec))
            / np.where(area == 0.0, 1.0, area))
    if not bool(np.all((area >= TOL) & (dist <= TOL))):
        flags["coplanar_ok"] = False

    if np.unique(cells).size != n_points:
        flags["ghost_ok"] = False

    fids = np.sort(cells[:, FACES].reshape(-1, 4), axis=1)
    order = np.lexsort((fids[:, 3], fids[:, 2], fids[:, 1], fids[:, 0]))
    srt = fids[order]
    new_key = np.empty(srt.shape[0], dtype=bool)
    new_key[0] = True
    new_key[1:] = np.any(srt[1:] != srt[:-1], axis=1)
    starts = np.flatnonzero(new_key)
    counts = np.diff(np.append(starts, srt.shape[0]))
    if not bool(np.all((counts == 1) | (counts == 2))):
        flags["share_ok"] = False

    boundary = srt[starts[counts == 1]]
    if boundary.size:
        bp = pts[boundary]                                # (B, 4, 3)
        x, y, z = bp[..., 0], bp[..., 1], bp[..., 2]
        eqx = (np.abs(x - CUBE_MIN) <= TOL) | (np.abs(x - CUBE_MAX) <= TOL)
        eqy = (np.abs(y - CUBE_MIN) <= TOL) | (np.abs(y - CUBE_MAX) <= TOL)
        eqz = (np.abs(z - CUBE_MIN) <= TOL) | (np.abs(z - CUBE_MAX) <= TOL)
        inx = (x >= CUBE_MIN - TOL) & (x <= CUBE_MAX + TOL)
        iny = (y >= CUBE_MIN - TOL) & (y <= CUBE_MAX + TOL)
        inz = (z >= CUBE_MIN - TOL) & (z <= CUBE_MAX + TOL)
        on_surface = ((eqx & iny & inz) | (eqy & inx & inz) | (eqz & inx & iny))
        if not bool(on_surface.all()):
            flags["boundary_ok"] = False

    if oct_pts is not None and not _point_coverage_ok(pts, oct_pts):
        flags["octree_ok"] = False

    return flags


def _analyze_grid_fallback(grid, budget: int, oct_pts: np.ndarray | None) -> dict[str, Any]:
    """Cell-by-cell path for meshes that are not pure hexahedra."""
    vtk, _ = _vtk_modules()
    flags = {k: True for k in FLAG_KEYS}
    points = np.ascontiguousarray(
        grid.GetPoints().GetData(), dtype=np.float64).reshape(-1, 3)
    n_cells = grid.GetNumberOfCells()
    n_points = grid.GetNumberOfPoints()
    if n_cells > budget:
        flags["count_ok"] = False
    used: set[int] = set()
    face_count: dict[tuple[int, ...], int] = {}
    for i in range(n_cells):
        cell = grid.GetCell(i)
        if cell.GetCellType() != vtk.VTK_HEXAHEDRON:
            flags["sj_ok"] = False
            flags["coplanar_ok"] = False
            flags["share_ok"] = False
            flags["boundary_ok"] = False
            for j in range(cell.GetNumberOfPoints()):
                used.add(cell.GetPointId(j))
            continue
        ids = [cell.GetPointId(j) for j in range(8)]
        used.update(ids)
        sub = _analyze_arrays(points, np.array([ids], dtype=np.int64), budget, None)
        if not sub["sj_ok"]:
            flags["sj_ok"] = False
        if not sub["coplanar_ok"]:
            flags["coplanar_ok"] = False
        for face in FACES:
            key = tuple(sorted(ids[k] for k in face))
            face_count[key] = face_count.get(key, 0) + 1
    if len(used) != n_points:
        flags["ghost_ok"] = False
    for key, cnt in face_count.items():
        if cnt not in (1, 2):
            flags["share_ok"] = False
            continue
        if cnt == 1:
            for pid in key:
                p = points[pid]
                if not _on_cube_surface(p):
                    flags["boundary_ok"] = False
                    break
    if oct_pts is not None and not _point_coverage_ok(points, oct_pts):
        flags["octree_ok"] = False
    return flags


def _on_cube_surface(p: np.ndarray) -> bool:
    x, y, z = (float(p[0]), float(p[1]), float(p[2]))
    def eq(v): return abs(v - CUBE_MIN) <= TOL or abs(v - CUBE_MAX) <= TOL
    def inr(v): return CUBE_MIN - TOL <= v <= CUBE_MAX + TOL
    return ((eq(x) and inr(y) and inr(z)) or (eq(y) and inr(x) and inr(z))
            or (eq(z) and inr(x) and inr(y)))


def _analyze_file(path: Path, budget: int, oct_pts: np.ndarray | None) -> dict[str, Any]:
    fail = {k: False for k in FLAG_KEYS}
    try:
        pts, cells = _read_hex_mesh(path)
        if pts is None:
            return dict(fail, reason=str(cells))
        if isinstance(cells, str):
            if cells in ("non_hexahedral_cells", "ragged_connectivity"):
                return dict(_analyze_grid_fallback(pts, budget, oct_pts), reason=cells)
            return dict(fail, reason=cells)
        return dict(_analyze_arrays(pts, cells, budget, oct_pts),
                    n_cells=int(cells.shape[0]), n_points=int(pts.shape[0]))
    except Exception as exc:
        return dict(fail, reason=f"{type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Instance-level collection (cached; shared by the three scorers)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_NAMES = tuple(f"{i}.vtk" for i in range(N_STATE_PAIRS))


def _score_of(path: Path) -> int:
    hits = sum(1 for name in REQUIRED_NAMES if (path / name).is_file())
    return hits


def _select_output_dir(pred_dir: Path) -> tuple[Path, int]:
    """Prefer the workspace root; otherwise the sub-directory with most outputs."""
    root_hits = _score_of(pred_dir)
    if root_hits == N_STATE_PAIRS:
        return pred_dir, root_hits
    best = (pred_dir, root_hits)
    if pred_dir.is_dir():
        for sub in sorted(p for p in pred_dir.rglob("*") if p.is_dir()):
            hits = _score_of(sub)
            if hits > best[1]:
                best = (sub, hits)
    return best


def _fingerprint(out_dir: Path) -> tuple[int, int, float]:
    n = size = 0
    latest = 0.0
    for name in REQUIRED_NAMES:
        p = out_dir / name
        try:
            st = p.stat()
        except OSError:
            continue
        n += 1
        size += st.st_size
        latest = max(latest, st.st_mtime)
    return (n, size, latest)


def analyze_instance(pred_dir: Path, ref_dir: Path) -> dict[str, Any]:
    """Analyse every required output file once and cache the result."""
    pred_dir = Path(pred_dir)
    ref_dir = Path(ref_dir)
    out_dir, hits = _select_output_dir(pred_dir)
    key = (str(out_dir), str(ref_dir), _fingerprint(out_dir))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    flags: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    n_cells_total = 0
    for i in range(N_STATE_PAIRS):
        path = out_dir / f"{i}.vtk"
        if not path.is_file():
            missing.append(i)
            flags[i] = {k: False for k in FLAG_KEYS}
            flags[i]["reason"] = "file_missing"
            continue
        oct_pts = _read_points(ref_dir / "octree" / f"{i}_ref.vtk")
        res = _analyze_file(path, THRES[i], oct_pts)
        if oct_pts is None:
            res["octree_ok"] = False
            res["reason"] = "reference_octree_missing"
        n_cells_total += int(res.get("n_cells", 0))
        flags[i] = res

    stray = sorted(
        p.name for p in out_dir.glob("*.vtk")
        if p.stem not in {str(i) for i in range(N_STATE_PAIRS)}
    )[:_MAX_REPORTED]

    report: dict[str, Any] = {
        "output_dir": str(out_dir),
        "output_dir_relative": (out_dir.relative_to(pred_dir).as_posix()
                                if out_dir != pred_dir else "."),
        "n_expected": N_STATE_PAIRS,
        "n_found": N_STATE_PAIRS - len(missing),
        "missing_indices": missing[:_MAX_REPORTED],
        "n_missing": len(missing),
        "stray_vtk_files": stray,
        "n_cells_total": n_cells_total,
        "flags": flags,
    }
    with _CACHE_LOCK:
        while len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = report
    return report


def _summarise(report: dict[str, Any], needed: tuple[str, ...]) -> tuple[bool, dict[str, Any], str]:
    failing = [i for i in range(N_STATE_PAIRS)
               if not all(report["flags"][i][f] for f in needed)]
    per_flag = {f: sum(1 for i in range(N_STATE_PAIRS) if not report["flags"][i][f])
                for f in needed}
    examples = [
        {"file": f"{i}.vtk",
         "failed": [f for f in needed if not report["flags"][i][f]],
         "reason": report["flags"][i].get("reason")}
        for i in failing[:_MAX_REPORTED]
    ]
    ok = not failing
    details = {
        "files_expected": report["n_expected"],
        "files_found": report["n_found"],
        "files_missing": report["n_missing"],
        "missing_examples": [f"{i}.vtk" for i in report["missing_indices"]],
        "scored_directory": report["output_dir_relative"],
        "stray_vtk_files": report["stray_vtk_files"],
        "files_failing_this_scorer": len(failing),
        "failures_per_check": per_flag,
        "failing_examples": examples,
    }
    if ok:
        message = ""
    elif report["n_missing"] == N_STATE_PAIRS:
        message = (f"no required output found (expected {N_STATE_PAIRS} files "
                   f"0.vtk..{N_STATE_PAIRS - 1}.vtk under {details['scored_directory']})")
    else:
        message = (f"{len(failing)}/{N_STATE_PAIRS} files fail "
                   f"({', '.join(f'{k}={v}' for k, v in per_flag.items() if v)}"
                   + (f"; {report['n_missing']} missing" if report["n_missing"] else "")
                   + f"; first: {examples[0]['file']} {examples[0]['failed']})")
    return ok, details, message


class _MeshScorer(Scorer):
    """Base class: all-or-nothing credit over all 2048 state pairs."""

    required_flags: tuple[str, ...] = ()

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        report = analyze_instance(Path(pred_dir), Path(ref_dir))
        ok, details, message = _summarise(report, self.required_flags)
        return ScoreDetail(
            scorer_name=self.name,
            score=weight if ok else 0.0,
            max_score=weight,
            passed=ok,
            details=details,
            message=message,
        )


@register_scorer("mesh_quality")
@register_scorer("mesh quality")
class MeshQualityScorer(_MeshScorer):
    """All cells hexahedral with minimum scaled Jacobian > 0.21."""
    required_flags = QUALITY_FLAGS


@register_scorer("coplanarity")
@register_scorer("mesh_coplanarity")
class CoplanarityScorer(_MeshScorer):
    """All four vertices of every face of every hex are coplanar (1e-6)."""
    required_flags = COPLANARITY_FLAGS


@register_scorer("mesh_consistency")
@register_scorer("mesh consistency")
class MeshConsistencyScorer(_MeshScorer):
    """Element budget, no ghost nodes, conforming sharing, boundary, node coverage."""
    required_flags = CONSISTENCY_FLAGS
