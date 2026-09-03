extern "C" {

struct ChromaprintStubContext {
    int algorithm;
};

const char* chromaprint_get_version(void) {
    return "compat-stub-1.0";
}

void* chromaprint_new(int algorithm) {
    static ChromaprintStubContext context = {};
    context.algorithm = algorithm;
    return &context;
}

void chromaprint_free(void* /*ctx*/) {}

int chromaprint_get_algorithm(void* ctx) {
    if (!ctx) {
        return 0;
    }
    return static_cast<ChromaprintStubContext*>(ctx)->algorithm;
}

int chromaprint_set_option(void* /*ctx*/, const char* /*name*/, int /*value*/) {
    return 0;
}

int chromaprint_start(void* /*ctx*/, int /*sample_rate*/, int /*num_channels*/) {
    return 0;
}

int chromaprint_feed(void* /*ctx*/, void* /*data*/, int /*size*/) {
    return 0;
}

int chromaprint_finish(void* /*ctx*/) {
    return 0;
}

int chromaprint_get_fingerprint(void* /*ctx*/, char** fingerprint) {
    if (fingerprint) {
        *fingerprint = nullptr;
    }
    return 0;
}

int chromaprint_get_raw_fingerprint(void* /*ctx*/, void** fingerprint, int* size) {
    if (fingerprint) {
        *fingerprint = nullptr;
    }
    if (size) {
        *size = 0;
    }
    return 0;
}

int chromaprint_get_fingerprint_hash(void* /*ctx*/, void* /*hash*/) {
    return 0;
}

int chromaprint_encode_fingerprint(const void* /*fp*/,
                                   int /*size*/,
                                   int /*algorithm*/,
                                   void** encoded_fp,
                                   int* encoded_size,
                                   int /*base64*/) {
    if (encoded_fp) {
        *encoded_fp = nullptr;
    }
    if (encoded_size) {
        *encoded_size = 0;
    }
    return 0;
}

int chromaprint_decode_fingerprint(const void* /*encoded_fp*/,
                                   int /*encoded_size*/,
                                   void** fp,
                                   int* size,
                                   int* algorithm,
                                   int /*base64*/) {
    if (fp) {
        *fp = nullptr;
    }
    if (size) {
        *size = 0;
    }
    if (algorithm) {
        *algorithm = 0;
    }
    return 0;
}

int chromaprint_hash_fingerprint(const void* /*fp*/, int /*size*/, void* /*hash*/) {
    return 0;
}

void chromaprint_dealloc(void* /*ptr*/) {}

}  // extern "C"
