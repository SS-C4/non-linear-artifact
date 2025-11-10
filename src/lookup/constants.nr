pub type FUNC_TYPE = super::structs::SoftMaxLookupWitness;
pub global NUM_INPUTS: u32 = 10;
pub global NUM_SOFTMAX: u32 = 10;
pub global LOG_S: u32 = 12;
pub global S : Field = 4096; // 2^12
pub global LOG_BASE: u32 = 4;
pub global BASE: u32 = 16;
pub global MULT_OPS: [(Field, Field, Field); 3] = [
    (0, 1, 4),
    (2, 3, 5),
    (4, 5, 6),
];
pub global NUM_TABLES: u32 = 4;
pub global BASE_POWERS: [Field; 4] = [
    1,
    16,
    256,
    4096,
];
pub global EXP_TABLES: [[Field; 16]; 4] = [
    [4096, 4095, 4094, 4093, 4092, 4091, 4090, 4089, 4088, 4087, 4086, 4085, 4084, 4083, 4082, 4081],
    [4096, 4080, 4064, 4048, 4032, 4017, 4001, 3986, 3970, 3955, 3939, 3924, 3908, 3893, 3878, 3863],
    [4096, 3848, 3615, 3396, 3190, 2997, 2815, 2645, 2484, 2334, 2192, 2060, 1935, 1818, 1707, 1604],
    [4096, 1507, 554, 204, 75, 28, 10, 4, 1, 1, 0, 0, 0, 0, 0, 0],
];
pub global COSINE_TABLE: [Field; 0] = [];
