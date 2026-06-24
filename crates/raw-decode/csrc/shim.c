// Thin C ABI over LibRaw. We do all libraw_data_t/params struct access here,
// in C, against the installed libraw.h — so Rust never depends on the exact
// (version-specific) struct layout. Rust talks to this stable 5-field result.
#include <libraw/libraw.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned short *data; // interleaved RGB, 16-bit per channel, row-major
    unsigned int width;
    unsigned int height;
    unsigned int channels; // always 3
    int error;             // 0 = ok, otherwise LibRaw error code
} aml_image;

// Decode a RAW file to linear (gamma 1.0), sRGB-primary, 16-bit RGB.
// We deliberately disable auto-brightening and keep linear data so the GPU
// develop pipeline is the single source of tone/exposure decisions.
aml_image aml_decode_linear(const char *path) {
    aml_image out;
    memset(&out, 0, sizeof(out));

    libraw_data_t *lr = libraw_init(0);
    if (!lr) { out.error = -1000; return out; }

    int rc;
    if ((rc = libraw_open_file(lr, path)) != LIBRAW_SUCCESS) { out.error = rc; goto done; }
    if ((rc = libraw_unpack(lr)) != LIBRAW_SUCCESS)          { out.error = rc; goto done; }

    lr->params.output_bps     = 16;   // 16-bit output
    lr->params.output_color   = 1;    // sRGB primaries
    lr->params.gamm[0]        = 1.0;  // linear transfer (gamma 1.0 / no toe)
    lr->params.gamm[1]        = 1.0;
    lr->params.no_auto_bright = 1;    // we own exposure
    lr->params.use_camera_wb  = 1;    // sane WB starting point
    lr->params.user_flip      = -1;   // honor embedded orientation

    if ((rc = libraw_dcraw_process(lr)) != LIBRAW_SUCCESS) { out.error = rc; goto done; }

    int err = 0;
    libraw_processed_image_t *img = libraw_dcraw_make_mem_image(lr, &err);
    if (!img || err != LIBRAW_SUCCESS) { out.error = err ? err : -1001; goto done; }
    if (img->type != LIBRAW_IMAGE_BITMAP || img->bits != 16 || img->colors != 3) {
        out.error = -1002;
        libraw_dcraw_clear_mem(img);
        goto done;
    }

    out.width    = img->width;
    out.height   = img->height;
    out.channels = img->colors;
    out.data     = (unsigned short *)malloc(img->data_size);
    if (!out.data) { out.error = -1003; libraw_dcraw_clear_mem(img); goto done; }
    memcpy(out.data, img->data, img->data_size);
    libraw_dcraw_clear_mem(img);

done:
    libraw_close(lr);
    return out;
}

void aml_free(unsigned short *data) { free(data); }

// Lightweight metadata probe for catalog import: open + parse headers only
// (no unpack/demosaic), so importing a folder is fast.
typedef struct {
    int error;
    unsigned int width;
    unsigned int height;
    int flip;            // libraw orientation flag
    long long timestamp; // capture time, unix seconds
    float iso;
    float shutter;       // seconds
    float aperture;      // f-number
    float focal;         // mm
    char make[64];
    char model[64];
    char lens[64];
} aml_meta;

aml_meta aml_probe(const char *path) {
    aml_meta m;
    memset(&m, 0, sizeof(m));

    libraw_data_t *lr = libraw_init(0);
    if (!lr) { m.error = -1000; return m; }

    int rc = libraw_open_file(lr, path);
    if (rc != LIBRAW_SUCCESS) { m.error = rc; libraw_close(lr); return m; }

    m.width     = lr->sizes.width;
    m.height    = lr->sizes.height;
    m.flip      = lr->sizes.flip;
    m.timestamp = (long long)lr->other.timestamp;
    m.iso       = lr->other.iso_speed;
    m.shutter   = lr->other.shutter;
    m.aperture  = lr->other.aperture;
    m.focal     = lr->other.focal_len;
    strncpy(m.make,  lr->idata.make,  sizeof(m.make) - 1);
    strncpy(m.model, lr->idata.model, sizeof(m.model) - 1);
    strncpy(m.lens,  lr->lens.Lens,   sizeof(m.lens) - 1);

    libraw_close(lr);
    return m;
}
