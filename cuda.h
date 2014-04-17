#ifndef CUDA_H
#define CUDA_H

#ifdef HAVE_CUDA
extern bool CUDA_active;
extern bool opt_reorder;
extern int opt_hysteresis;
extern int opt_targettemp;
extern int opt_overheattemp;
void init_CUDA(int nDevs);
float gpu_temp(int gpu);
int gpu_engineclock(int gpu);
int gpu_memclock(int gpu);
float gpu_vddc(int gpu);
int gpu_activity(int gpu);
int gpu_fanspeed(int gpu);
int gpu_fanpercent(int gpu);
bool gpu_stats(int gpu, float *temp, int *engineclock, int *memclock, float *vddc,
	       int *activity, int *fanspeed, int *fanpercent, int *powertune);
void change_gpusettings(int gpu);
void gpu_autotune(int gpu, enum dev_enable *denable);
void clear_CUDA(int nDevs);

#else /* HAVE_CUDA */

#define CUDA_active (0)
static inline void init_CUDA(__maybe_unused int nDevs) {}
static inline void change_gpusettings(__maybe_unused int gpu) { }
static inline void clear_CUDA(__maybe_unused int nDevs) {}

#endif /* HAVE_CUDA */

#endif /* CUDA_H */
