#include "config.h"

#if defined(HAVE_CUDA) && (defined(__unix__) || defined (WIN32))

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_CURSES
#	include <curses.h>
#endif

#include "miner.h"
#include "CUDA_SDK/CUDA_sdk.h"
#include "compat.h"

#if defined (__unix__)
#	include <dlfcn.h>
#	include <stdlib.h>
#	include <unistd.h>
#else /* WIN32 */
#	include <windows.h>
#	include <tchar.h>
#endif
#include "CUDA_functions.h"

#ifndef HAVE_CURSES
#	define wlogprint(...)  applog(LOG_WARNING, __VA_ARGS__)
#endif

bool CUDA_active;
bool opt_reorder = false;

int opt_hysteresis = 3;
int opt_targettemp = 75;
int opt_overheattemp = 85;
static pthread_mutex_t CUDA_lock;

struct gpu_adapters {
	int iAdapterIndex;
	int iBusNumber;
	int virtual_gpu;
	int id;
};

// Memory allocation function
static void * __stdcall CUDA_Main_Memory_Alloc(int iSize)
{
	void *lpBuffer = malloc(iSize);

	return lpBuffer;
}

// Optional Memory de-allocation function
static void __stdcall CUDA_Main_Memory_Free (void **lpBuffer)
{
	if (*lpBuffer != NULL) {
		free (*lpBuffer);
		*lpBuffer = NULL;
	}
}

#if defined (UNIX)
// equivalent functions in linux
static void *GetProcAddress(void *pLibrary, const char *name)
{
	return dlsym( pLibrary, name);
}
#endif

static	CUDA_MAIN_CONTROL_CREATE			CUDA_Main_Control_Create;
static	CUDA_MAIN_CONTROL_DESTROY		CUDA_Main_Control_Destroy;
static	CUDA_ADAPTER_NUMBEROFADAPTERS_GET	CUDA_Adapter_NumberOfAdapters_Get;
static	CUDA_ADAPTER_ADAPTERINFO_GET		CUDA_Adapter_AdapterInfo_Get;
static	CUDA_ADAPTER_ID_GET			CUDA_Adapter_ID_Get;
static	CUDA_MAIN_CONTROL_REFRESH		CUDA_Main_Control_Refresh;
static	CUDA_ADAPTER_VIDEOBIOSINFO_GET		CUDA_Adapter_VideoBiosInfo_Get;
static	CUDA_DISPLAY_DISPLAYINFO_GET		CUDA_Display_DisplayInfo_Get;
static	CUDA_ADAPTER_ACCESSIBILITY_GET		CUDA_Adapter_Accessibility_Get;

static	CUDA_OVERDRIVE_CAPS			CUDA_Overdrive_Caps;

static	CUDA_OVERDRIVE5_TEMPERATURE_GET		CUDA_Overdrive5_Temperature_Get;
static	CUDA_OVERDRIVE5_CURRENTACTIVITY_GET	CUDA_Overdrive5_CurrentActivity_Get;
static	CUDA_OVERDRIVE5_ODPARAMETERS_GET		CUDA_Overdrive5_ODParameters_Get;
static	CUDA_OVERDRIVE5_FANSPEEDINFO_GET		CUDA_Overdrive5_FanSpeedInfo_Get;
static	CUDA_OVERDRIVE5_FANSPEED_GET		CUDA_Overdrive5_FanSpeed_Get;
static	CUDA_OVERDRIVE5_FANSPEED_SET		CUDA_Overdrive5_FanSpeed_Set;
static	CUDA_OVERDRIVE5_ODPERFORMANCELEVELS_GET	CUDA_Overdrive5_ODPerformanceLevels_Get;
static	CUDA_OVERDRIVE5_ODPERFORMANCELEVELS_SET	CUDA_Overdrive5_ODPerformanceLevels_Set;
static	CUDA_OVERDRIVE5_POWERCONTROL_GET		CUDA_Overdrive5_PowerControl_Get;
static	CUDA_OVERDRIVE5_POWERCONTROL_SET		CUDA_Overdrive5_PowerControl_Set;
static	CUDA_OVERDRIVE5_FANSPEEDTODEFAULT_SET	CUDA_Overdrive5_FanSpeedToDefault_Set;

static	CUDA_OVERDRIVE6_CAPABILITIES_GET		CUDA_Overdrive6_Capabilities_Get;
static	CUDA_OVERDRIVE6_FANSPEED_GET		CUDA_Overdrive6_FanSpeed_Get;
static	CUDA_OVERDRIVE6_THERMALCONTROLLER_CAPS	CUDA_Overdrive6_ThermalController_Caps;
static	CUDA_OVERDRIVE6_TEMPERATURE_GET		CUDA_Overdrive6_Temperature_Get;
static	CUDA_OVERDRIVE6_STATEINFO_GET		CUDA_Overdrive6_StateInfo_Get;
static	CUDA_OVERDRIVE6_CURRENTSTATUS_GET	CUDA_Overdrive6_CurrentStatus_Get;
static	CUDA_OVERDRIVE6_POWERCONTROL_CAPS	CUDA_Overdrive6_PowerControl_Caps;
static	CUDA_OVERDRIVE6_POWERCONTROLINFO_GET	CUDA_Overdrive6_PowerControlInfo_Get;
static	CUDA_OVERDRIVE6_POWERCONTROL_GET		CUDA_Overdrive6_PowerControl_Get;
static	CUDA_OVERDRIVE6_FANSPEED_SET		CUDA_Overdrive6_FanSpeed_Set;
static	CUDA_OVERDRIVE6_STATE_SET		CUDA_Overdrive6_State_Set;
static	CUDA_OVERDRIVE6_POWERCONTROL_SET		CUDA_Overdrive6_PowerControl_Set;

#if defined (UNIX)
	static void *hDLL;	// Handle to .so library
#else
	HINSTANCE hDLL;		// Handle to DLL
#endif
static int iNumberAdapters;
static LPAdapterInfo lpInfo = NULL;

int set_fanspeed(int gpu, int iFanSpeed);
static float __gpu_temp(struct gpu_CUDA *ga);

char *CUDA_error_desc(int error)
{
	char *result;
	switch(error)
	{
		case CUDA_ERR:
			result = "Generic error (escape call failed?)";
			break;
		case CUDA_ERR_NOT_INIT:
			result = "CUDA not initialized";
			break;
		case CUDA_ERR_INVALID_PARAM:
			result = "Invalid parameter";
			break;
		case CUDA_ERR_INVALID_PARAM_SIZE:
			result = "Invalid parameter size";
			break;
		case CUDA_ERR_INVALID_CUDA_IDX:
			result = "Invalid CUDA index";
			break;
		case CUDA_ERR_INVALID_CONTROLLER_IDX:
			result = "Invalid controller index";
			break;
		case CUDA_ERR_INVALID_DIPLAY_IDX:
			result = "Invalid display index";
			break;
		case CUDA_ERR_NOT_SUPPORTED:
			result = "Function not supported by the driver";
			break;
		case CUDA_ERR_NULL_POINTER:
			result = "Null Pointer error";
			break;
		case CUDA_ERR_DISABLED_ADAPTER:
			result = "Disabled adapter, can't make call";
			break;
		case CUDA_ERR_INVALID_CALLBACK:
			result = "Invalid callback";
			break;
		case CUDA_ERR_RESOURCE_CONFLICT:
			result = "Display resource conflict";
			break;
		case CUDA_ERR_SET_INCOMPLETE:
			result = "Failed to update some of the values";
			break;
		case CUDA_ERR_NO_XDISPLAY:
			result = "No Linux XDisplay in Linux Console environment";
			break;
		default:
			result = "Unhandled error";
			break;
	}
	return result;
}

static inline void lock_CUDA(void)
{
	mutex_lock(&CUDA_lock);
}

static inline void unlock_CUDA(void)
{
	mutex_unlock(&CUDA_lock);
}

/* This looks for the twin GPU that has the fanspeed control of a non fanspeed
 * control GPU on dual GPU cards */
static bool fanspeed_twin(struct gpu_CUDA *ga, struct gpu_CUDA *other_ga)
{
	if (!other_ga->has_fanspeed)
		return false;
	if (abs(ga->iBusNumber - other_ga->iBusNumber) != 1)
		return false;
	if (strcmp(ga->strAdapterName, other_ga->strAdapterName))
		return false;
	return true;
}

static bool init_overdrive5()
{
	CUDA_Overdrive5_Temperature_Get = (CUDA_OVERDRIVE5_TEMPERATURE_GET) GetProcAddress(hDLL,"CUDA_Overdrive5_Temperature_Get");
	CUDA_Overdrive5_CurrentActivity_Get = (CUDA_OVERDRIVE5_CURRENTACTIVITY_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_CurrentActivity_Get");
	CUDA_Overdrive5_ODParameters_Get = (CUDA_OVERDRIVE5_ODPARAMETERS_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_ODParameters_Get");
	CUDA_Overdrive5_FanSpeedInfo_Get = (CUDA_OVERDRIVE5_FANSPEEDINFO_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_FanSpeedInfo_Get");
	CUDA_Overdrive5_FanSpeed_Get = (CUDA_OVERDRIVE5_FANSPEED_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_FanSpeed_Get");
	CUDA_Overdrive5_FanSpeed_Set = (CUDA_OVERDRIVE5_FANSPEED_SET) GetProcAddress(hDLL, "CUDA_Overdrive5_FanSpeed_Set");
	CUDA_Overdrive5_ODPerformanceLevels_Get = (CUDA_OVERDRIVE5_ODPERFORMANCELEVELS_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_ODPerformanceLevels_Get");
	CUDA_Overdrive5_ODPerformanceLevels_Set = (CUDA_OVERDRIVE5_ODPERFORMANCELEVELS_SET) GetProcAddress(hDLL, "CUDA_Overdrive5_ODPerformanceLevels_Set");
	CUDA_Overdrive5_PowerControl_Get = (CUDA_OVERDRIVE5_POWERCONTROL_GET) GetProcAddress(hDLL, "CUDA_Overdrive5_PowerControl_Get");
	CUDA_Overdrive5_PowerControl_Set = (CUDA_OVERDRIVE5_POWERCONTROL_SET) GetProcAddress(hDLL, "CUDA_Overdrive5_PowerControl_Set");
	CUDA_Overdrive5_FanSpeedToDefault_Set = (CUDA_OVERDRIVE5_FANSPEEDTODEFAULT_SET) GetProcAddress(hDLL, "CUDA_Overdrive5_FanSpeedToDefault_Set");

	if (!CUDA_Overdrive5_Temperature_Get || !CUDA_Overdrive5_CurrentActivity_Get ||
		!CUDA_Overdrive5_ODParameters_Get || !CUDA_Overdrive5_FanSpeedInfo_Get ||
		!CUDA_Overdrive5_FanSpeed_Get || !CUDA_Overdrive5_FanSpeed_Set ||
		!CUDA_Overdrive5_ODPerformanceLevels_Get || !CUDA_Overdrive5_ODPerformanceLevels_Set ||
		!CUDA_Overdrive5_PowerControl_Get ||	!CUDA_Overdrive5_PowerControl_Set ||
		!CUDA_Overdrive5_FanSpeedToDefault_Set) {
			applog(LOG_WARNING, "ATI CUDA Overdrive5's API is missing or broken.");
			return false;
	} else {
		applog(LOG_INFO, "ATI CUDA Overdrive5 API found.");
	}

	return true;
}

static bool init_overdrive6()
{
	CUDA_Overdrive6_FanSpeed_Get = (CUDA_OVERDRIVE6_FANSPEED_GET) GetProcAddress(hDLL,"CUDA_Overdrive6_FanSpeed_Get");
	CUDA_Overdrive6_ThermalController_Caps = (CUDA_OVERDRIVE6_THERMALCONTROLLER_CAPS)GetProcAddress (hDLL, "CUDA_Overdrive6_ThermalController_Caps");
	CUDA_Overdrive6_Temperature_Get = (CUDA_OVERDRIVE6_TEMPERATURE_GET)GetProcAddress (hDLL, "CUDA_Overdrive6_Temperature_Get");
	CUDA_Overdrive6_Capabilities_Get = (CUDA_OVERDRIVE6_CAPABILITIES_GET)GetProcAddress(hDLL, "CUDA_Overdrive6_Capabilities_Get");
	CUDA_Overdrive6_StateInfo_Get = (CUDA_OVERDRIVE6_STATEINFO_GET)GetProcAddress(hDLL, "CUDA_Overdrive6_StateInfo_Get");
	CUDA_Overdrive6_CurrentStatus_Get = (CUDA_OVERDRIVE6_CURRENTSTATUS_GET)GetProcAddress(hDLL, "CUDA_Overdrive6_CurrentStatus_Get");
	CUDA_Overdrive6_PowerControl_Caps = (CUDA_OVERDRIVE6_POWERCONTROL_CAPS)GetProcAddress(hDLL, "CUDA_Overdrive6_PowerControl_Caps");
	CUDA_Overdrive6_PowerControlInfo_Get = (CUDA_OVERDRIVE6_POWERCONTROLINFO_GET)GetProcAddress(hDLL, "CUDA_Overdrive6_PowerControlInfo_Get");
	CUDA_Overdrive6_PowerControl_Get = (CUDA_OVERDRIVE6_POWERCONTROL_GET)GetProcAddress(hDLL, "CUDA_Overdrive6_PowerControl_Get");
	CUDA_Overdrive6_FanSpeed_Set  = (CUDA_OVERDRIVE6_FANSPEED_SET)GetProcAddress(hDLL, "CUDA_Overdrive6_FanSpeed_Set");
	CUDA_Overdrive6_State_Set = (CUDA_OVERDRIVE6_STATE_SET)GetProcAddress(hDLL, "CUDA_Overdrive6_State_Set");
	CUDA_Overdrive6_PowerControl_Set = (CUDA_OVERDRIVE6_POWERCONTROL_SET) GetProcAddress(hDLL, "CUDA_Overdrive6_PowerControl_Set");

	if (!CUDA_Overdrive6_FanSpeed_Get || !CUDA_Overdrive6_ThermalController_Caps ||
		!CUDA_Overdrive6_Temperature_Get || !CUDA_Overdrive6_Capabilities_Get ||
		!CUDA_Overdrive6_StateInfo_Get || !CUDA_Overdrive6_CurrentStatus_Get ||
		!CUDA_Overdrive6_PowerControl_Caps || !CUDA_Overdrive6_PowerControlInfo_Get ||
		!CUDA_Overdrive6_PowerControl_Get || !CUDA_Overdrive6_FanSpeed_Set ||
		!CUDA_Overdrive6_State_Set || !CUDA_Overdrive6_PowerControl_Set) {
			applog(LOG_WARNING, "ATI CUDA Overdrive6's API is missing or broken.");
		return false;
	} else {
		applog(LOG_INFO, "ATI CUDA Overdrive6 API found.");
	}

	return true;
}

static bool prepare_CUDA(void)
{
	int result;

#if defined (UNIX)
	hDLL = dlopen( "libatiCUDAxx.so", RTLD_LAZY|RTLD_GLOBAL);
#else
	hDLL = LoCUDAibrary("atiCUDAxx.dll");
	if (hDLL == NULL)
		// A 32 bit calling application on 64 bit OS will fail to LoCUDAIbrary.
		// Try to load the 32 bit library (atiCUDAxy.dll) instead
		hDLL = LoCUDAibrary("atiCUDAxy.dll");
#endif
	if (hDLL == NULL) {
		applog(LOG_INFO, "Unable to load ATI CUDA library.");
		return false;
	}
	CUDA_Main_Control_Create = (CUDA_MAIN_CONTROL_CREATE) GetProcAddress(hDLL,"CUDA_Main_Control_Create");
	CUDA_Main_Control_Destroy = (CUDA_MAIN_CONTROL_DESTROY) GetProcAddress(hDLL,"CUDA_Main_Control_Destroy");
	CUDA_Adapter_NumberOfAdapters_Get = (CUDA_ADAPTER_NUMBEROFADAPTERS_GET) GetProcAddress(hDLL,"CUDA_Adapter_NumberOfAdapters_Get");
	CUDA_Adapter_AdapterInfo_Get = (CUDA_ADAPTER_ADAPTERINFO_GET) GetProcAddress(hDLL,"CUDA_Adapter_AdapterInfo_Get");
	CUDA_Display_DisplayInfo_Get = (CUDA_DISPLAY_DISPLAYINFO_GET) GetProcAddress(hDLL,"CUDA_Display_DisplayInfo_Get");
	CUDA_Adapter_ID_Get = (CUDA_ADAPTER_ID_GET) GetProcAddress(hDLL,"CUDA_Adapter_ID_Get");
	CUDA_Main_Control_Refresh = (CUDA_MAIN_CONTROL_REFRESH) GetProcAddress(hDLL, "CUDA_Main_Control_Refresh");
	CUDA_Adapter_VideoBiosInfo_Get = (CUDA_ADAPTER_VIDEOBIOSINFO_GET)GetProcAddress(hDLL,"CUDA_Adapter_VideoBiosInfo_Get");
	CUDA_Overdrive_Caps = (CUDA_OVERDRIVE_CAPS)GetProcAddress(hDLL, "CUDA_Overdrive_Caps");

	CUDA_Adapter_Accessibility_Get = (CUDA_ADAPTER_ACCESSIBILITY_GET)GetProcAddress(hDLL, "CUDA_Adapter_Accessibility_Get");

	if (!CUDA_Main_Control_Create || !CUDA_Main_Control_Destroy ||
		!CUDA_Adapter_NumberOfAdapters_Get || !CUDA_Adapter_AdapterInfo_Get ||
		!CUDA_Display_DisplayInfo_Get ||
		!CUDA_Adapter_ID_Get || !CUDA_Main_Control_Refresh ||
		!CUDA_Adapter_VideoBiosInfo_Get || !CUDA_Overdrive_Caps) {
			applog(LOG_WARNING, "ATI CUDA API is missing or broken.");
		return false;
	}

	// Initialise CUDA. The second parameter is 1, which means:
	// retrieve adapter information only for adapters that are physically present and enabled in the system
	result = CUDA_Main_Control_Create(CUDA_Main_Memory_Alloc, 1);
	if (result != CUDA_OK) {
		applog(LOG_INFO, "CUDA initialisation error: %d (%s)", result, CUDA_error_desc(result));
		return false;
	}

	result = CUDA_Main_Control_Refresh();
	if (result != CUDA_OK) {
		applog(LOG_INFO, "CUDA refresh error: %d (%s)", result, CUDA_error_desc(result));
		return false;
	}

	init_overdrive5();
	init_overdrive6(); // FIXME: don't if CUDA6 is not present

	return true;
}

void init_CUDA(int nDevs)
{
	int result, i, j, devices = 0, last_adapter = -1, gpu = 0, dummy = 0;
	struct gpu_adapters adapters[MAX_GPUDEVICES], vadapters[MAX_GPUDEVICES];
	bool devs_match = true;
	CUDABiosInfo BiosInfo;

	applog(LOG_INFO, "Number of CUDA devices: %d", nDevs);

	if (unlikely(pthread_mutex_init(&CUDA_lock, NULL))) {
		applog(LOG_ERR, "Failed to init CUDA_lock in init_CUDA");
		return;
	}

	if (!prepare_CUDA())
		return;

	// Obtain the number of adapters for the system
	result = CUDA_Adapter_NumberOfAdapters_Get (&iNumberAdapters);
	if (result != CUDA_OK) {
		applog(LOG_INFO, "Cannot get the number of adapters! Error %d!", result);
		return ;
	}

	if (iNumberAdapters > 0) {
		lpInfo = (LPAdapterInfo)malloc ( sizeof (AdapterInfo) * iNumberAdapters );
		memset ( lpInfo,'\0', sizeof (AdapterInfo) * iNumberAdapters );

		lpInfo->iSize = sizeof(lpInfo);
		// Get the AdapterInfo structure for all adapters in the system
		result = CUDA_Adapter_AdapterInfo_Get (lpInfo, sizeof (AdapterInfo) * iNumberAdapters);
		if (result != CUDA_OK) {
			applog(LOG_INFO, "CUDA_Adapter_AdapterInfo_Get Error! Error %d", result);
			return ;
		}
	} else {
		applog(LOG_INFO, "No adapters found");
		return;
	}

	applog(LOG_INFO, "Found %d logical CUDA adapters", iNumberAdapters);

	/* Iterate over iNumberAdapters and find the lpAdapterID of real devices */
	for (i = 0; i < iNumberAdapters; i++) {
		int iAdapterIndex;
		int lpAdapterID;

		iAdapterIndex = lpInfo[i].iAdapterIndex;

		/* Get unique identifier of the adapter, 0 means not AMD */
		result = CUDA_Adapter_ID_Get(iAdapterIndex, &lpAdapterID);

		if (CUDA_Adapter_VideoBiosInfo_Get(iAdapterIndex, &BiosInfo) == CUDA_ERR) {
			applog(LOG_INFO, "CUDA index %d, id %d - FAILED to get BIOS info", iAdapterIndex, lpAdapterID);
		} else {
			applog(LOG_INFO, "CUDA index %d, id %d - BIOS partno.: %s, version: %s, date: %s", iAdapterIndex, lpAdapterID, BiosInfo.strPartNumber, BiosInfo.strVersion, BiosInfo.strDate);
		}

		if (result != CUDA_OK) {
			applog(LOG_INFO, "Failed to CUDA_Adapter_ID_Get. Error %d", result);
			if (result == -10)
				applog(LOG_INFO, "(Device is not enabled.)");
			continue;
		}

		/* Each adapter may have multiple entries */
		if (lpAdapterID == last_adapter) {
			continue;
		}

		applog(LOG_INFO, "GPU %d assigned: "
		       "iAdapterIndex:%d "
		       "iPresent:%d "
		       "strUDID:%s "
		       "iBusNumber:%d "
		       "iDeviceNumber:%d "
#if defined(__linux__)
		       "iDrvIndex:%d "
#endif
		       "iFunctionNumber:%d "
		       "iVendorID:%d "
		       "name:%s",
		       devices,
		       lpInfo[i].iAdapterIndex,
		       lpInfo[i].iPresent,
		       lpInfo[i].strUDID,
		       lpInfo[i].iBusNumber,
		       lpInfo[i].iDeviceNumber,
#if defined(__linux__)
		       lpInfo[i].iDrvIndex,
#endif
		       lpInfo[i].iFunctionNumber,
		       lpInfo[i].iVendorID,
		       lpInfo[i].strAdapterName);

		adapters[devices].iAdapterIndex = iAdapterIndex;
		adapters[devices].iBusNumber = lpInfo[i].iBusNumber;
		adapters[devices].id = i;

		/* We found a truly new adapter instead of a logical
		 * one. Now since there's no way of correlating the
		 * opencl enumerated devices and the CUDA enumerated
		 * ones, we have to assume they're in the same order.*/
		if (++devices > nDevs && devs_match) {
			applog(LOG_ERR, "CUDA found more devices than opencl!");
			applog(LOG_ERR, "There is possibly at least one GPU that doesn't support OpenCL");
			applog(LOG_ERR, "Use the gpu map feature to reliably map OpenCL to CUDA");
			devs_match = false;
		}
		last_adapter = lpAdapterID;

		if (!lpAdapterID) {
			applog(LOG_INFO, "Adapter returns ID 0 meaning not AMD. Card order might be confused");
			continue;
		}
	}

	if (devices < nDevs) {
		applog(LOG_ERR, "CUDA found less devices than opencl!");
		applog(LOG_ERR, "There is possibly more than one display attached to a GPU");
		applog(LOG_ERR, "Use the gpu map feature to reliably map OpenCL to CUDA");
		devs_match = false;
	}

	for (i = 0; i < devices; i++) {
		vadapters[i].virtual_gpu = i;
		vadapters[i].id = adapters[i].id;
	}

	/* Apply manually provided OpenCL to CUDA mapping, if any */
	for (i = 0; i < nDevs; i++) {
		if (gpus[i].mapped) {
			vadapters[gpus[i].virtual_CUDA].virtual_gpu = i;
			applog(LOG_INFO, "Mapping OpenCL device %d to CUDA device %d", i, gpus[i].virtual_CUDA);
		} else {
			gpus[i].virtual_CUDA = i;
		}
	}

	if (!devs_match) {
		applog(LOG_ERR, "WARNING: Number of OpenCL and CUDA devices did not match!");
		applog(LOG_ERR, "Hardware monitoring may NOT match up with devices!");
	} else if (opt_reorder) {
		/* Windows has some kind of random ordering for bus number IDs and
		 * ordering the GPUs according to ascending order fixes it. Linux
		 * has usually sequential but decreasing order instead! */
		for (i = 0; i < devices; i++) {
			int j, virtual_gpu;

			virtual_gpu = 0;
			for (j = 0; j < devices; j++) {
				if (i == j)
					continue;
#ifdef WIN32
				if (adapters[j].iBusNumber < adapters[i].iBusNumber)
#else
				if (adapters[j].iBusNumber > adapters[i].iBusNumber)
#endif
					virtual_gpu++;
			}
			if (virtual_gpu != i) {
				applog(LOG_INFO, "Mapping device %d to GPU %d according to Bus Number order",
				       i, virtual_gpu);
				vadapters[virtual_gpu].virtual_gpu = i;
				vadapters[virtual_gpu].id = adapters[i].id;
			}
		}
	}

	if (devices > nDevs)
		devices = nDevs;

	for (gpu = 0; gpu < devices; gpu++) {
		struct gpu_CUDA *ga;
		int iAdapterIndex;
		int lpAdapterID;
		CUDAODPerformanceLevels *lpOdPerformanceLevels;
		int lev, CUDAGpu;
		size_t plsize;
		CUDABiosInfo BiosInfo;

		CUDAGpu = gpus[gpu].virtual_CUDA;
		i = vadapters[CUDAGpu].id;
		iAdapterIndex = lpInfo[i].iAdapterIndex;
		gpus[gpu].virtual_gpu = vadapters[CUDAGpu].virtual_gpu;

		/* Get unique identifier of the adapter, 0 means not AMD */
		result = CUDA_Adapter_ID_Get(iAdapterIndex, &lpAdapterID);
		if (result != CUDA_OK) {
			applog(LOG_INFO, "Failed to CUDA_Adapter_ID_Get. Error %d", result);
			continue;
		}

		if (gpus[gpu].deven == DEV_DISABLED) {
			gpus[gpu].gpu_engine =
			gpus[gpu].gpu_memclock =
			gpus[gpu].gpu_vddc =
			gpus[gpu].gpu_fan =
			gpus[gpu].gpu_powertune = 0;
			continue;
		}

		applog(LOG_INFO, "GPU %d %s hardware monitoring enabled", gpu, lpInfo[i].strAdapterName);
		if (gpus[gpu].name)
			free(gpus[gpu].name);
		gpus[gpu].name = lpInfo[i].strAdapterName;
		gpus[gpu].has_CUDA = true;
		/* Flag CUDA as active if any card is successfully activated */
		CUDA_active = true;

		/* From here on we know this device is a discrete device and
		 * should support CUDA */
		ga = &gpus[gpu].CUDA;
		ga->gpu = gpu;
		ga->iAdapterIndex = iAdapterIndex;
		ga->lpAdapterID = lpAdapterID;
		strcpy(ga->strAdapterName, lpInfo[i].strAdapterName);
		ga->DefPerfLev = NULL;
		ga->twin = NULL;
		ga->def_fan_valid = false;

		applog(LOG_INFO, "CUDA GPU %d is Adapter index %d and maps to adapter id %d", ga->gpu, ga->iAdapterIndex, ga->lpAdapterID);

		if (CUDA_Adapter_VideoBiosInfo_Get(iAdapterIndex, &BiosInfo) != CUDA_ERR)
			applog(LOG_INFO, "GPU %d BIOS partno.: %s, version: %s, date: %s", gpu, BiosInfo.strPartNumber, BiosInfo.strVersion, BiosInfo.strDate);

		ga->lpOdParameters.iSize = sizeof(CUDAODParameters);
		if (CUDA_Overdrive5_ODParameters_Get(iAdapterIndex, &ga->lpOdParameters) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_ODParameters_Get");

		lev = ga->lpOdParameters.iNumberOfPerformanceLevels - 1;
		/* We're only interested in the top performance level */
		plsize = sizeof(CUDAODPerformanceLevels) + lev * sizeof(CUDAODPerformanceLevel);
		lpOdPerformanceLevels = (CUDAODPerformanceLevels *)malloc(plsize);
		lpOdPerformanceLevels->iSize = plsize;

		/* Get default performance levels first */
		if (CUDA_Overdrive5_ODPerformanceLevels_Get(iAdapterIndex, 1, lpOdPerformanceLevels) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_ODPerformanceLevels_Get");
		/* Set the limits we'd use based on default gpu speeds */
		ga->maxspeed = ga->minspeed = lpOdPerformanceLevels->aLevels[lev].iEngineClock;

		ga->lpTemperature.iSize = sizeof(CUDATemperature);
		ga->lpFanSpeedInfo.iSize = sizeof(CUDAFanSpeedInfo);
		ga->lpFanSpeedValue.iSize = ga->DefFanSpeedValue.iSize = sizeof(CUDAFanSpeedValue);
		ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_RPM;
		ga->DefFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_RPM;

		/* Now get the current performance levels for any existing overclock */
		if (CUDA_Overdrive5_ODPerformanceLevels_Get(iAdapterIndex, 0, lpOdPerformanceLevels) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_ODPerformanceLevels_Get");
		else {
			/* Save these values as the defaults in case we wish to reset to defaults */
			ga->DefPerfLev = (CUDAODPerformanceLevels *)malloc(plsize);
			memcpy(ga->DefPerfLev, lpOdPerformanceLevels, plsize);
		}

		if (gpus[gpu].gpu_engine) {
			int setengine = gpus[gpu].gpu_engine * 100;

			/* Lower profiles can't have a higher setting */
			for (j = 0; j < lev; j++) {
				if (lpOdPerformanceLevels->aLevels[j].iEngineClock > setengine)
					lpOdPerformanceLevels->aLevels[j].iEngineClock = setengine;
			}
			lpOdPerformanceLevels->aLevels[lev].iEngineClock = setengine;
			applog(LOG_INFO, "Setting GPU %d engine clock to %d", gpu, gpus[gpu].gpu_engine);
			CUDA_Overdrive5_ODPerformanceLevels_Set(iAdapterIndex, lpOdPerformanceLevels);
			ga->maxspeed = setengine;
			if (gpus[gpu].min_engine)
				ga->minspeed = gpus[gpu].min_engine * 100;
			ga->managed = true;
			if (gpus[gpu].gpu_memdiff)
				set_memoryclock(gpu, gpus[gpu].gpu_engine + gpus[gpu].gpu_memdiff);
		}

		if (gpus[gpu].gpu_memclock) {
			int setmem = gpus[gpu].gpu_memclock * 100;

			for (j = 0; j < lev; j++) {
				if (lpOdPerformanceLevels->aLevels[j].iMemoryClock > setmem)
					lpOdPerformanceLevels->aLevels[j].iMemoryClock = setmem;
			}
			lpOdPerformanceLevels->aLevels[lev].iMemoryClock = setmem;
			applog(LOG_INFO, "Setting GPU %d memory clock to %d", gpu, gpus[gpu].gpu_memclock);
			CUDA_Overdrive5_ODPerformanceLevels_Set(iAdapterIndex, lpOdPerformanceLevels);
			ga->managed = true;
		}

		if (gpus[gpu].gpu_vddc) {
			int setv = gpus[gpu].gpu_vddc * 1000;

			for (j = 0; j < lev; j++) {
				if (lpOdPerformanceLevels->aLevels[j].iVddc > setv)
					lpOdPerformanceLevels->aLevels[j].iVddc = setv;
			}
			lpOdPerformanceLevels->aLevels[lev].iVddc = setv;
			applog(LOG_INFO, "Setting GPU %d voltage to %.3f", gpu, gpus[gpu].gpu_vddc);
			CUDA_Overdrive5_ODPerformanceLevels_Set(iAdapterIndex, lpOdPerformanceLevels);
			ga->managed = true;
		}

		CUDA_Overdrive5_ODPerformanceLevels_Get(iAdapterIndex, 0, lpOdPerformanceLevels);
		ga->iEngineClock = lpOdPerformanceLevels->aLevels[lev].iEngineClock;
		ga->iMemoryClock = lpOdPerformanceLevels->aLevels[lev].iMemoryClock;
		ga->iVddc = lpOdPerformanceLevels->aLevels[lev].iVddc;
		ga->iBusNumber = lpInfo[i].iBusNumber;

		if (CUDA_Overdrive5_FanSpeedInfo_Get(iAdapterIndex, 0, &ga->lpFanSpeedInfo) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_FanSpeedInfo_Get");

		if(!(ga->lpFanSpeedInfo.iFlags & (CUDA_DL_FANCTRL_SUPPORTS_RPM_WRITE | CUDA_DL_FANCTRL_SUPPORTS_PERCENT_WRITE)))
			ga->has_fanspeed = false;
		else
			ga->has_fanspeed = true;

		/* Save the fanspeed values as defaults in case we reset later */
		if (CUDA_Overdrive5_FanSpeed_Get(iAdapterIndex, 0, &ga->DefFanSpeedValue) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_FanSpeed_Get for default value");
		else
			ga->def_fan_valid = true;

		if (gpus[gpu].gpu_fan)
			set_fanspeed(gpu, gpus[gpu].gpu_fan);
		else
			gpus[gpu].gpu_fan = 85; /* Set a nominal upper limit of 85% */

		/* Not fatal if powercontrol get fails */
		if (CUDA_Overdrive5_PowerControl_Get(iAdapterIndex, &ga->iPercentage, &dummy) != CUDA_OK)
			applog(LOG_INFO, "Failed to CUDA_Overdrive5_PowerControl_get");

		if (gpus[gpu].gpu_powertune) {
			CUDA_Overdrive5_PowerControl_Set(iAdapterIndex, gpus[gpu].gpu_powertune);
			CUDA_Overdrive5_PowerControl_Get(iAdapterIndex, &ga->iPercentage, &dummy);
			ga->managed = true;
		}

		/* Set some default temperatures for autotune when enabled */
		if (!ga->targettemp)
			ga->targettemp = opt_targettemp;
		if (!ga->overtemp)
			ga->overtemp = opt_overheattemp;
		if (!gpus[gpu].cutofftemp)
			gpus[gpu].cutofftemp = opt_cutofftemp;
		if (opt_autofan) {
			/* Set a safe starting default if we're automanaging fan speeds */
			int nominal = 50;

			ga->autofan = true;
			/* Clamp fanspeed values to range provided */
			if (nominal > gpus[gpu].gpu_fan)
				nominal = gpus[gpu].gpu_fan;
			if (nominal < gpus[gpu].min_fan)
				nominal = gpus[gpu].min_fan;
			set_fanspeed(gpu, nominal);
		}
		if (opt_autoengine) {
			ga->autoengine = true;
			ga->managed = true;
		}
		ga->lasttemp = __gpu_temp(ga);
	}

	for (gpu = 0; gpu < devices; gpu++) {
		struct gpu_CUDA *ga = &gpus[gpu].CUDA;
		int j;

		for (j = 0; j < devices; j++) {
			struct gpu_CUDA *other_ga;

			if (j == gpu)
				continue;

			other_ga = &gpus[j].CUDA;

			/* Search for twin GPUs on a single card. They will be
			 * separated by one bus id and one will have fanspeed
			 * while the other won't. */
			if (!ga->has_fanspeed) {
				if (fanspeed_twin(ga, other_ga)) {
					applog(LOG_INFO, "Dual GPUs detected: %d and %d",
						ga->gpu, other_ga->gpu);
					ga->twin = other_ga;
					other_ga->twin = ga;
				}
			}
		}
	}
}

static float __gpu_temp(struct gpu_CUDA *ga)
{
	if (CUDA_Overdrive5_Temperature_Get(ga->iAdapterIndex, 0, &ga->lpTemperature) != CUDA_OK)
		return -1;
	return (float)ga->lpTemperature.iTemperature / 1000;
}

float gpu_temp(int gpu)
{
	struct gpu_CUDA *ga;
	float ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	ret = __gpu_temp(ga);
	unlock_CUDA();
	gpus[gpu].temp = ret;
	return ret;
}

static inline int __gpu_engineclock(struct gpu_CUDA *ga)
{
	return ga->lpActivity.iEngineClock / 100;
}

int gpu_engineclock(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	if (CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity) != CUDA_OK)
		goto out;
	ret = __gpu_engineclock(ga);
out:
	unlock_CUDA();
	return ret;
}

static inline int __gpu_memclock(struct gpu_CUDA *ga)
{
	return ga->lpActivity.iMemoryClock / 100;
}

int gpu_memclock(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	if (CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity) != CUDA_OK)
		goto out;
	ret = __gpu_memclock(ga);
out:
	unlock_CUDA();
	return ret;
}

static inline float __gpu_vddc(struct gpu_CUDA *ga)
{
	return (float)ga->lpActivity.iVddc / 1000;
}

float gpu_vddc(int gpu)
{
	struct gpu_CUDA *ga;
	float ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	if (CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity) != CUDA_OK)
		goto out;
	ret = __gpu_vddc(ga);
out:
	unlock_CUDA();
	return ret;
}

static inline int __gpu_activity(struct gpu_CUDA *ga)
{
	if (!ga->lpOdParameters.iActivityReportingSupported)
		return -1;
	return ga->lpActivity.iActivityPercent;
}

int gpu_activity(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	ret = CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity);
	unlock_CUDA();
	if (ret != CUDA_OK)
		return ret;
	if (!ga->lpOdParameters.iActivityReportingSupported)
		return ret;
	return ga->lpActivity.iActivityPercent;
}

static inline int __gpu_fanspeed(struct gpu_CUDA *ga)
{
	if (!ga->has_fanspeed && ga->twin)
		return __gpu_fanspeed(ga->twin);

	if (!(ga->lpFanSpeedInfo.iFlags & CUDA_DL_FANCTRL_SUPPORTS_RPM_READ))
		return -1;
	ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_RPM;
	if (CUDA_Overdrive5_FanSpeed_Get(ga->iAdapterIndex, 0, &ga->lpFanSpeedValue) != CUDA_OK)
		return -1;
	return ga->lpFanSpeedValue.iFanSpeed;
}

int gpu_fanspeed(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	ret = __gpu_fanspeed(ga);
	unlock_CUDA();
	return ret;
}

static int __gpu_fanpercent(struct gpu_CUDA *ga)
{
	if (!ga->has_fanspeed && ga->twin)
		return __gpu_fanpercent(ga->twin);

	if (!(ga->lpFanSpeedInfo.iFlags & CUDA_DL_FANCTRL_SUPPORTS_PERCENT_READ ))
		return -1;
	ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_PERCENT;
	if (CUDA_Overdrive5_FanSpeed_Get(ga->iAdapterIndex, 0, &ga->lpFanSpeedValue) != CUDA_OK)
		return -1;
	return ga->lpFanSpeedValue.iFanSpeed;
}

int gpu_fanpercent(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	ret = __gpu_fanpercent(ga);
	unlock_CUDA();
	if (unlikely(ga->has_fanspeed && ret == -1)) {
#if 0
		/* Recursive calling applog causes a hang, so disable messages */
		applog(LOG_WARNING, "GPU %d stopped reporting fanspeed due to driver corruption", gpu);
		if (opt_restart) {
			applog(LOG_WARNING, "Restart enabled, will attempt to restart sgminer");
			applog(LOG_WARNING, "You can disable this with the --no-restart option");
			app_restart();
		}
		applog(LOG_WARNING, "Disabling fanspeed monitoring on this device");
		ga->has_fanspeed = false;
		if (ga->twin) {
			applog(LOG_WARNING, "Disabling fanspeed linking on GPU twins");
			ga->twin->twin = NULL;;
			ga->twin = NULL;
		}
#endif
		if (opt_restart)
			app_restart();
		ga->has_fanspeed = false;
		if (ga->twin) {
			ga->twin->twin = NULL;;
			ga->twin = NULL;
		}
	}
	return ret;
}

static inline int __gpu_powertune(struct gpu_CUDA *ga)
{
	int dummy = 0;

	if (CUDA_Overdrive5_PowerControl_Get(ga->iAdapterIndex, &ga->iPercentage, &dummy) != CUDA_OK)
		return -1;
	return ga->iPercentage;
}

int gpu_powertune(int gpu)
{
	struct gpu_CUDA *ga;
	int ret = -1;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return ret;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	ret = __gpu_powertune(ga);
	unlock_CUDA();
	return ret;
}

bool gpu_stats(int gpu, float *temp, int *engineclock, int *memclock, float *vddc,
	       int *activity, int *fanspeed, int *fanpercent, int *powertune)
{
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return false;

	ga = &gpus[gpu].CUDA;

	lock_CUDA();
	*temp = __gpu_temp(ga);
	if (CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity) != CUDA_OK) {
		*engineclock = 0;
		*memclock = 0;
		*vddc = 0;
		*activity = 0;
	} else {
		*engineclock = __gpu_engineclock(ga);
		*memclock = __gpu_memclock(ga);
		*vddc = __gpu_vddc(ga);
		*activity = __gpu_activity(ga);
	}
	*fanspeed = __gpu_fanspeed(ga);
	*fanpercent = __gpu_fanpercent(ga);
	*powertune = __gpu_powertune(ga);
	unlock_CUDA();

	return true;
}

#ifdef HAVE_CURSES
static void get_enginerange(int gpu, int *imin, int *imax)
{
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Get enginerange not supported\n");
		return;
	}
	ga = &gpus[gpu].CUDA;
	*imin = ga->lpOdParameters.sEngineClock.iMin / 100;
	*imax = ga->lpOdParameters.sEngineClock.iMax / 100;
}
#endif

int set_engineclock(int gpu, int iEngineClock)
{
	CUDAODPerformanceLevels *lpOdPerformanceLevels;
	struct cgpu_info *cgpu;
	int i, lev, ret = 1;
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Set engineclock not supported\n");
		return ret;
	}

	iEngineClock *= 100;
	ga = &gpus[gpu].CUDA;

	/* Keep track of intended engine clock in case the device changes
	 * profile and drops while idle, not taking the new engine clock */
	ga->lastengine = iEngineClock;

	lev = ga->lpOdParameters.iNumberOfPerformanceLevels - 1;
	lpOdPerformanceLevels = (CUDAODPerformanceLevels *)alloca(sizeof(CUDAODPerformanceLevels)+(lev * sizeof(CUDAODPerformanceLevel)));
	lpOdPerformanceLevels->iSize = sizeof(CUDAODPerformanceLevels) + sizeof(CUDAODPerformanceLevel) * lev;

	lock_CUDA();
	if (CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels) != CUDA_OK)
		goto out;
	for (i = 0; i < lev; i++) {
		if (lpOdPerformanceLevels->aLevels[i].iEngineClock > iEngineClock)
			lpOdPerformanceLevels->aLevels[i].iEngineClock = iEngineClock;
	}
	lpOdPerformanceLevels->aLevels[lev].iEngineClock = iEngineClock;
	CUDA_Overdrive5_ODPerformanceLevels_Set(ga->iAdapterIndex, lpOdPerformanceLevels);
	CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels);
	if (lpOdPerformanceLevels->aLevels[lev].iEngineClock == iEngineClock)
		ret = 0;
	ga->iEngineClock = lpOdPerformanceLevels->aLevels[lev].iEngineClock;
	if (ga->iEngineClock > ga->maxspeed)
		ga->maxspeed = ga->iEngineClock;
	if (ga->iEngineClock < ga->minspeed)
		ga->minspeed = ga->iEngineClock;
	ga->iMemoryClock = lpOdPerformanceLevels->aLevels[lev].iMemoryClock;
	ga->iVddc = lpOdPerformanceLevels->aLevels[lev].iVddc;
	ga->managed = true;
out:
	unlock_CUDA();

	cgpu = &gpus[gpu];
	if (cgpu->gpu_memdiff)
		set_memoryclock(gpu, iEngineClock / 100 + cgpu->gpu_memdiff);

	return ret;
}

#ifdef HAVE_CURSES
static void get_memoryrange(int gpu, int *imin, int *imax)
{
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Get memoryrange not supported\n");
		return;
	}
	ga = &gpus[gpu].CUDA;
	*imin = ga->lpOdParameters.sMemoryClock.iMin / 100;
	*imax = ga->lpOdParameters.sMemoryClock.iMax / 100;
}
#endif

int set_memoryclock(int gpu, int iMemoryClock)
{
	CUDAODPerformanceLevels *lpOdPerformanceLevels;
	int i, lev, ret = 1;
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Set memoryclock not supported\n");
		return ret;
	}

	iMemoryClock *= 100;
	ga = &gpus[gpu].CUDA;

	lev = ga->lpOdParameters.iNumberOfPerformanceLevels - 1;
	lpOdPerformanceLevels = (CUDAODPerformanceLevels *)alloca(sizeof(CUDAODPerformanceLevels)+(lev * sizeof(CUDAODPerformanceLevel)));
	lpOdPerformanceLevels->iSize = sizeof(CUDAODPerformanceLevels) + sizeof(CUDAODPerformanceLevel) * lev;

	lock_CUDA();
	if (CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels) != CUDA_OK)
		goto out;
	lpOdPerformanceLevels->aLevels[lev].iMemoryClock = iMemoryClock;
	for (i = 0; i < lev; i++) {
		if (lpOdPerformanceLevels->aLevels[i].iMemoryClock > iMemoryClock)
			lpOdPerformanceLevels->aLevels[i].iMemoryClock = iMemoryClock;
	}
	CUDA_Overdrive5_ODPerformanceLevels_Set(ga->iAdapterIndex, lpOdPerformanceLevels);
	CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels);
	if (lpOdPerformanceLevels->aLevels[lev].iMemoryClock == iMemoryClock)
		ret = 0;
	ga->iEngineClock = lpOdPerformanceLevels->aLevels[lev].iEngineClock;
	ga->iMemoryClock = lpOdPerformanceLevels->aLevels[lev].iMemoryClock;
	ga->iVddc = lpOdPerformanceLevels->aLevels[lev].iVddc;
	ga->managed = true;
out:
	unlock_CUDA();
	return ret;
}

#ifdef HAVE_CURSES
static void get_vddcrange(int gpu, float *imin, float *imax)
{
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Get vddcrange not supported\n");
		return;
	}
	ga = &gpus[gpu].CUDA;
	*imin = (float)ga->lpOdParameters.sVddc.iMin / 1000;
	*imax = (float)ga->lpOdParameters.sVddc.iMax / 1000;
}

static float curses_float(const char *query)
{
	float ret;
	char *cvar;

	cvar = curses_input(query);
	ret = atof(cvar);
	free(cvar);
	return ret;
}
#endif

int set_vddc(int gpu, float fVddc)
{
	CUDAODPerformanceLevels *lpOdPerformanceLevels;
	int i, iVddc, lev, ret = 1;
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Set vddc not supported\n");
		return ret;
	}

	iVddc = 1000 * fVddc;
	ga = &gpus[gpu].CUDA;

	lev = ga->lpOdParameters.iNumberOfPerformanceLevels - 1;
	lpOdPerformanceLevels = (CUDAODPerformanceLevels *)alloca(sizeof(CUDAODPerformanceLevels)+(lev * sizeof(CUDAODPerformanceLevel)));
	lpOdPerformanceLevels->iSize = sizeof(CUDAODPerformanceLevels) + sizeof(CUDAODPerformanceLevel) * lev;

	lock_CUDA();
	if (CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels) != CUDA_OK)
		goto out;
	for (i = 0; i < lev; i++) {
		if (lpOdPerformanceLevels->aLevels[i].iVddc > iVddc)
			lpOdPerformanceLevels->aLevels[i].iVddc = iVddc;
	}
	lpOdPerformanceLevels->aLevels[lev].iVddc = iVddc;
	CUDA_Overdrive5_ODPerformanceLevels_Set(ga->iAdapterIndex, lpOdPerformanceLevels);
	CUDA_Overdrive5_ODPerformanceLevels_Get(ga->iAdapterIndex, 0, lpOdPerformanceLevels);
	if (lpOdPerformanceLevels->aLevels[lev].iVddc == iVddc)
		ret = 0;
	ga->iEngineClock = lpOdPerformanceLevels->aLevels[lev].iEngineClock;
	ga->iMemoryClock = lpOdPerformanceLevels->aLevels[lev].iMemoryClock;
	ga->iVddc = lpOdPerformanceLevels->aLevels[lev].iVddc;
	ga->managed = true;
out:
	unlock_CUDA();
	return ret;
}

static void get_fanrange(int gpu, int *imin, int *imax)
{
	struct gpu_CUDA *ga;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Get fanrange not supported\n");
		return;
	}
	ga = &gpus[gpu].CUDA;
	*imin = ga->lpFanSpeedInfo.iMinPercent;
	*imax = ga->lpFanSpeedInfo.iMaxPercent;
}

int set_fanspeed(int gpu, int iFanSpeed)
{
	struct gpu_CUDA *ga;
	int ret = 1;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Set fanspeed not supported\n");
		return ret;
	}

	ga = &gpus[gpu].CUDA;
	if (!(ga->lpFanSpeedInfo.iFlags & (CUDA_DL_FANCTRL_SUPPORTS_RPM_WRITE | CUDA_DL_FANCTRL_SUPPORTS_PERCENT_WRITE ))) {
		applog(LOG_DEBUG, "GPU %d doesn't support rpm or percent write", gpu);
		return ret;
	}

	/* Store what fanspeed we're actually aiming for for re-entrant changes
	 * in case this device does not support fine setting changes */
	ga->targetfan = iFanSpeed;

	lock_CUDA();
	ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_RPM;
	if (CUDA_Overdrive5_FanSpeed_Get(ga->iAdapterIndex, 0, &ga->lpFanSpeedValue) != CUDA_OK) {
		applog(LOG_DEBUG, "GPU %d call to fanspeed get failed", gpu);
	}
	if (!(ga->lpFanSpeedValue.iFlags & CUDA_DL_FANCTRL_FLAG_USER_DEFINED_SPEED)) {
		/* If user defined is not already specified, set it first */
		ga->lpFanSpeedValue.iFlags |= CUDA_DL_FANCTRL_FLAG_USER_DEFINED_SPEED;
		CUDA_Overdrive5_FanSpeed_Set(ga->iAdapterIndex, 0, &ga->lpFanSpeedValue);
	}
	if (!(ga->lpFanSpeedInfo.iFlags & CUDA_DL_FANCTRL_SUPPORTS_PERCENT_WRITE)) {
		/* Must convert speed to an RPM */
		iFanSpeed = ga->lpFanSpeedInfo.iMaxRPM * iFanSpeed / 100;
		ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_RPM;
	} else
		ga->lpFanSpeedValue.iSpeedType = CUDA_DL_FANCTRL_SPEED_TYPE_PERCENT;
	ga->lpFanSpeedValue.iFanSpeed = iFanSpeed;
	ret = CUDA_Overdrive5_FanSpeed_Set(ga->iAdapterIndex, 0, &ga->lpFanSpeedValue);
	ga->managed = true;
	unlock_CUDA();

	return ret;
}

#ifdef HAVE_CURSES
static int set_powertune(int gpu, int iPercentage)
{
	struct gpu_CUDA *ga;
	int dummy, ret = 1;

	if (!gpus[gpu].has_CUDA || !CUDA_active) {
		wlogprint("Set powertune not supported\n");
		return ret;
	}

	ga = &gpus[gpu].CUDA;

	lock_CUDA();
	CUDA_Overdrive5_PowerControl_Set(ga->iAdapterIndex, iPercentage);
	CUDA_Overdrive5_PowerControl_Get(ga->iAdapterIndex, &ga->iPercentage, &dummy);
	if (ga->iPercentage == iPercentage)
		ret = 0;
	ga->managed = true;
	unlock_CUDA();
	return ret;
}
#endif

/* Returns whether the fanspeed is optimal already or not. The fan_window bool
 * tells us whether the current fanspeed is in the target range for fanspeeds.
 */
static bool fan_autotune(int gpu, int temp, int fanpercent, int lasttemp, bool *fan_window)
{
	struct cgpu_info *cgpu = &gpus[gpu];
	int tdiff = round((double)(temp - lasttemp));
	struct gpu_CUDA *ga = &cgpu->CUDA;
	int top = gpus[gpu].gpu_fan;
	int bot = gpus[gpu].min_fan;
	int newpercent = fanpercent;
	int iMin = 0, iMax = 100;

	get_fanrange(gpu, &iMin, &iMax);
	if (temp > ga->overtemp && fanpercent < iMax) {
		applog(LOG_WARNING, "Overheat detected on GPU %d, increasing fan to 100%%", gpu);
		newpercent = iMax;

		dev_error(cgpu, REASON_DEV_OVER_HEAT);
	} else if (temp > ga->targettemp && fanpercent < top && tdiff >= 0) {
		applog(LOG_DEBUG, "Temperature over target, increasing fanspeed");
		if (temp > ga->targettemp + opt_hysteresis)
			newpercent = ga->targetfan + 10;
		else
			newpercent = ga->targetfan + 5;
		if (newpercent > top)
			newpercent = top;
	} else if (fanpercent > bot && temp < ga->targettemp - opt_hysteresis) {
		/* Detect large swings of 5 degrees or more and change fan by
		 * a proportion more */
		if (tdiff <= 0) {
			applog(LOG_DEBUG, "Temperature %d degrees below target, decreasing fanspeed", opt_hysteresis);
			newpercent = ga->targetfan - 1 + tdiff / 5;
		} else if (tdiff >= 5) {
			applog(LOG_DEBUG, "Temperature climbed %d while below target, increasing fanspeed", tdiff);
			newpercent = ga->targetfan + tdiff / 5;
		}
	} else {

		/* We're in the optimal range, make minor adjustments if the
		 * temp is still drifting */
		if (fanpercent > bot && tdiff < 0 && lasttemp < ga->targettemp) {
			applog(LOG_DEBUG, "Temperature dropping while in target range, decreasing fanspeed");
			newpercent = ga->targetfan + tdiff;
		} else if (fanpercent < top && tdiff > 0 && temp > ga->targettemp - opt_hysteresis) {
			applog(LOG_DEBUG, "Temperature rising while in target range, increasing fanspeed");
			newpercent = ga->targetfan + tdiff;
		}
	}

	if (newpercent > iMax)
		newpercent = iMax;
	else if (newpercent < iMin)
		newpercent = iMin;

	if (newpercent <= top)
		*fan_window = true;
	else
		*fan_window = false;

	if (newpercent != fanpercent) {
		applog(LOG_INFO, "Setting GPU %d fan percentage to %d", gpu, newpercent);
		set_fanspeed(gpu, newpercent);
		/* If the fanspeed is going down and we're below the top speed,
		 * consider the fan optimal to prevent minute changes in
		 * fanspeed delaying GPU engine speed changes */
		if (newpercent < fanpercent && *fan_window)
			return true;
		return false;
	}
	return true;
}

void gpu_autotune(int gpu, enum dev_enable *denable)
{
	int temp, fanpercent, engine, newengine, twintemp = 0;
	bool fan_optimal = true, fan_window = true;
	struct cgpu_info *cgpu;
	struct gpu_CUDA *ga;

	cgpu = &gpus[gpu];
	ga = &cgpu->CUDA;

	lock_CUDA();
	CUDA_Overdrive5_CurrentActivity_Get(ga->iAdapterIndex, &ga->lpActivity);
	temp = __gpu_temp(ga);
	if (ga->twin)
		twintemp = __gpu_temp(ga->twin);
	fanpercent = __gpu_fanpercent(ga);
	unlock_CUDA();

	newengine = engine = gpu_engineclock(gpu) * 100;

	if (temp && fanpercent >= 0 && ga->autofan) {
		if (!ga->twin)
			fan_optimal = fan_autotune(gpu, temp, fanpercent, ga->lasttemp, &fan_window);
		else if (ga->autofan && (ga->has_fanspeed || !ga->twin->autofan)) {
			/* On linked GPUs, we autotune the fan only once, based
			 * on the highest temperature from either GPUs */
			int hightemp, fan_gpu;
			int lasttemp;

			if (twintemp > temp) {
				lasttemp = ga->twin->lasttemp;
				hightemp = twintemp;
			} else {
				lasttemp = ga->lasttemp;
				hightemp = temp;
			}
			if (ga->has_fanspeed)
				fan_gpu = gpu;
			else
				fan_gpu = ga->twin->gpu;
			fan_optimal = fan_autotune(fan_gpu, hightemp, fanpercent, lasttemp, &fan_window);
		}
	}

	if (engine && ga->autoengine) {
		if (temp > cgpu->cutofftemp) {
			applog(LOG_WARNING, "Hit thermal cutoff limit on GPU %d, disabling!", gpu);
			*denable = DEV_RECOVER;
			newengine = ga->minspeed;
			dev_error(cgpu, REASON_DEV_THERMAL_CUTOFF);
		} else if (temp > ga->overtemp && engine > ga->minspeed) {
			applog(LOG_WARNING, "Overheat detected, decreasing GPU %d clock speed", gpu);
			newengine = ga->minspeed;

			dev_error(cgpu, REASON_DEV_OVER_HEAT);
		} else if (temp > ga->targettemp + opt_hysteresis && engine > ga->minspeed && fan_optimal) {
			applog(LOG_DEBUG, "Temperature %d degrees over target, decreasing clock speed", opt_hysteresis);
			newengine = engine - ga->lpOdParameters.sEngineClock.iStep;
			/* Only try to tune engine speed up if this GPU is not disabled */
		} else if (temp < ga->targettemp && engine < ga->maxspeed && fan_window && *denable == DEV_ENABLED) {
			int iStep = ga->lpOdParameters.sEngineClock.iStep;

			applog(LOG_DEBUG, "Temperature below target, increasing clock speed");
			if (temp < ga->targettemp - opt_hysteresis)
				iStep *= 2;
			newengine = engine + iStep;
		} else if (temp < ga->targettemp && *denable == DEV_RECOVER && opt_restart) {
			applog(LOG_NOTICE, "Device recovered to temperature below target, re-enabling");
			*denable = DEV_ENABLED;
		}

		if (newengine > ga->maxspeed)
			newengine = ga->maxspeed;
		else if (newengine < ga->minspeed)
			newengine = ga->minspeed;

		/* Adjust engine clock speed if it's lower, or if it's higher
		 * but higher than the last intended value as well as the
		 * current speed, to avoid setting the engine clock speed to
		 * a speed relateive to a lower profile during idle periods. */
		if (newengine < engine || (newengine > engine && newengine > ga->lastengine)) {
			newengine /= 100;
			applog(LOG_INFO, "Setting GPU %d engine clock to %d", gpu, newengine);
			set_engineclock(gpu, newengine);
		}
	}
	ga->lasttemp = temp;
}

void set_defaultfan(int gpu)
{
	struct gpu_CUDA *ga;
	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	if (ga->def_fan_valid)
		CUDA_Overdrive5_FanSpeed_Set(ga->iAdapterIndex, 0, &ga->DefFanSpeedValue);
	unlock_CUDA();
}

void set_defaultengine(int gpu)
{
	struct gpu_CUDA *ga;
	if (!gpus[gpu].has_CUDA || !CUDA_active)
		return;

	ga = &gpus[gpu].CUDA;
	lock_CUDA();
	if (ga->DefPerfLev)
		CUDA_Overdrive5_ODPerformanceLevels_Set(ga->iAdapterIndex, ga->DefPerfLev);
	unlock_CUDA();
}

#ifdef HAVE_CURSES
void change_autosettings(int gpu)
{
	struct gpu_CUDA *ga = &gpus[gpu].CUDA;
	char input;
	int val;

	wlogprint("Target temperature: %d\n", ga->targettemp);
	wlogprint("Overheat temperature: %d\n", ga->overtemp);
	wlogprint("Cutoff temperature: %d\n", gpus[gpu].cutofftemp);
	wlogprint("Toggle [F]an auto [G]PU auto\nChange [T]arget [O]verheat [C]utoff\n");
	wlogprint("Or press any other key to continue\n");
	input = getch();
	if (!strncasecmp(&input, "f", 1)) {
		ga->autofan ^= true;
		wlogprint("Fan autotune is now %s\n", ga->autofan ? "enabled" : "disabled");
		if (!ga->autofan) {
			wlogprint("Resetting fan to startup settings\n");
			set_defaultfan(gpu);
		}
	} else if (!strncasecmp(&input, "g", 1)) {
		ga->autoengine ^= true;
		wlogprint("GPU engine clock autotune is now %s\n", ga->autoengine ? "enabled" : "disabled");
		if (!ga->autoengine) {
			wlogprint("Resetting GPU engine clock to startup settings\n");
			set_defaultengine(gpu);
		}
	} else if (!strncasecmp(&input, "t", 1)) {
		val = curses_int("Enter target temperature for this GPU in C (0-200)");
		if (val < 0 || val > 200)
			wlogprint("Invalid temperature");
		else
			ga->targettemp = val;
	} else if (!strncasecmp(&input, "o", 1)) {
		wlogprint("Enter overheat temperature for this GPU in C (%d+)", ga->targettemp);
		val = curses_int("");
		if (val <= ga->targettemp || val > 200)
			wlogprint("Invalid temperature");
		else
			ga->overtemp = val;
	} else if (!strncasecmp(&input, "c", 1)) {
		wlogprint("Enter cutoff temperature for this GPU in C (%d+)", ga->overtemp);
		val = curses_int("");
		if (val <= ga->overtemp || val > 200)
			wlogprint("Invalid temperature");
		else
			gpus[gpu].cutofftemp = val;
	}
}

void change_gpusettings(int gpu)
{
	struct gpu_CUDA *ga = &gpus[gpu].CUDA;
	float fval, fmin = 0, fmax = 0;
	int val, imin = 0, imax = 0;
	char input;
	int engineclock = 0, memclock = 0, activity = 0, fanspeed = 0, fanpercent = 0, powertune = 0;
	float temp = 0, vddc = 0;

updated:
	if (gpu_stats(gpu, &temp, &engineclock, &memclock, &vddc, &activity, &fanspeed, &fanpercent, &powertune))
	wlogprint("Temp: %.1f C\n", temp);
	if (fanpercent >= 0 || fanspeed >= 0) {
		wlogprint("Fan Speed: ");
		if (fanpercent >= 0)
			wlogprint("%d%% ", fanpercent);
		if (fanspeed >= 0)
			wlogprint("(%d RPM)", fanspeed);
		wlogprint("\n");
	}
	wlogprint("Engine Clock: %d MHz\nMemory Clock: %d Mhz\nVddc: %.3f V\nActivity: %d%%\nPowertune: %d%%\n",
		engineclock, memclock, vddc, activity, powertune);
	wlogprint("Fan autotune is %s (%d-%d)\n", ga->autofan ? "enabled" : "disabled",
		  gpus[gpu].min_fan, gpus[gpu].gpu_fan);
	wlogprint("GPU engine clock autotune is %s (%d-%d)\n", ga->autoengine ? "enabled" : "disabled",
		ga->minspeed / 100, ga->maxspeed / 100);
	wlogprint("Change [A]utomatic [E]ngine [F]an [M]emory [V]oltage [P]owertune\n");
	wlogprint("Or press any other key to continue\n");
	input = getch();

	if (!strncasecmp(&input, "a", 1)) {
		change_autosettings(gpu);
	} else if (!strncasecmp(&input, "e", 1)) {
		get_enginerange(gpu, &imin, &imax);
		wlogprint("Enter GPU engine clock speed (%d - %d Mhz)", imin, imax);
		val = curses_int("");
		if (val < imin || val > imax) {
			wlogprint("Value is outside safe range, are you sure?\n");
			input = getch();
			if (strncasecmp(&input, "y", 1))
				return;
		}
		if (!set_engineclock(gpu, val))
			wlogprint("Driver reports success but check values below\n");
		else
			wlogprint("Failed to modify engine clock speed\n");
	} else if (!strncasecmp(&input, "f", 1)) {
		get_fanrange(gpu, &imin, &imax);
		wlogprint("Enter fan percentage (%d - %d %%)", imin, imax);
		val = curses_int("");
		if (val < imin || val > imax) {
			wlogprint("Value is outside safe range, are you sure?\n");
			input = getch();
			if (strncasecmp(&input, "y", 1))
				return;
		}
		if (!set_fanspeed(gpu, val))
			wlogprint("Driver reports success but check values below\n");
		else
			wlogprint("Failed to modify fan speed\n");
	} else if (!strncasecmp(&input, "m", 1)) {
		get_memoryrange(gpu, &imin, &imax);
		wlogprint("Enter GPU memory clock speed (%d - %d Mhz)", imin, imax);
		val = curses_int("");
		if (val < imin || val > imax) {
			wlogprint("Value is outside safe range, are you sure?\n");
			input = getch();
			if (strncasecmp(&input, "y", 1))
				return;
		}
		if (!set_memoryclock(gpu, val))
			wlogprint("Driver reports success but check values below\n");
		else
			wlogprint("Failed to modify memory clock speed\n");
	} else if (!strncasecmp(&input, "v", 1)) {
		get_vddcrange(gpu, &fmin, &fmax);
		wlogprint("Enter GPU voltage (%.3f - %.3f V)", fmin, fmax);
		fval = curses_float("");
		if (fval < fmin || fval > fmax) {
			wlogprint("Value is outside safe range, are you sure?\n");
			input = getch();
			if (strncasecmp(&input, "y", 1))
				return;
		}
		if (!set_vddc(gpu, fval))
			wlogprint("Driver reports success but check values below\n");
		else
			wlogprint("Failed to modify voltage\n");
	} else if (!strncasecmp(&input, "p", 1)) {
		val = curses_int("Enter powertune value (-20 - 20)");
		if (val < -20 || val > 20) {
			wlogprint("Value is outside safe range, are you sure?\n");
			input = getch();
			if (strncasecmp(&input, "y", 1))
				return;
		}
		if (!set_powertune(gpu, val))
			wlogprint("Driver reports success but check values below\n");
		else
			wlogprint("Failed to modify powertune value\n");
	} else {
		clear_logwin();
		return;
	}
	cgsleep_ms(1000);
	goto updated;
}
#endif

static void free_CUDA(void)
{
	CUDA_Main_Memory_Free ((void **)&lpInfo);
	CUDA_Main_Control_Destroy ();
#if defined (UNIX)
	dlclose(hDLL);
#else
	FreeLibrary(hDLL);
#endif
}

void clear_CUDA(int nDevs)
{
	struct gpu_CUDA *ga;
	int i;

	if (!CUDA_active)
		return;

	lock_CUDA();
	/* Try to reset values to their defaults */
	for (i = 0; i < nDevs; i++) {
		ga = &gpus[i].CUDA;
		/*  Only reset the values if we've changed them at any time */
		if (!gpus[i].has_CUDA || !ga->managed || !ga->DefPerfLev)
			continue;
		CUDA_Overdrive5_ODPerformanceLevels_Set(ga->iAdapterIndex, ga->DefPerfLev);
		free(ga->DefPerfLev);
		if (ga->def_fan_valid)
			CUDA_Overdrive5_FanSpeed_Set(ga->iAdapterIndex, 0, &ga->DefFanSpeedValue);
		CUDA_Overdrive5_FanSpeedToDefault_Set(ga->iAdapterIndex, 0);
	}
	CUDA_active = false;
	unlock_CUDA();
	free_CUDA();
}
#endif /* HAVE_CUDA */
