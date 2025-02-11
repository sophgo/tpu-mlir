#ifndef BMCPU_DEV_H
#define BMCPU_DEV_H

#define BMCPU_DEV_LOGLEVEL_DEBUG  0
#define BMCPU_DEV_LOGLEVEL_INFO   1
#define BMCPU_DEV_LOGLEVEL_WARN   2
#define BMCPU_DEV_LOGLEVEL_ERROR  3
#define BMCPU_DEV_LOGLEVEL_FATAL  4
#define BMCPU_DEV_LOGLEVEL_OFF    5

#ifdef __cplusplus
extern "C" {
#endif

void bmcpu_dev_log(unsigned int level, const char *format, ...);
void bmcpu_dev_flush_dcache(void *start, void *end);
void bmcpu_dev_flush_and_invalidate_dcache(void *start, void *end);

#ifdef __cplusplus
}
#endif

#endif   /* BMCPU_DEV_H */
