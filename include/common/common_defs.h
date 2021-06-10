
/*******************************************************************************************************************************
 *
 * Copyright (c) 2020 ModalAI, Inc.
 *
 ******************************************************************************************************************************/
#ifndef COMMON_DEFS
#define COMMON_DEFS

#define COMPILE_ASSERT(x, y) static_assert(x, y)
#define PADDING_DISABLED __attribute__((packed))

static const int INT_INVALID_VALUE = 0xdeadbeef;
static const int MAX_NAME_LENGTH   = 128;
static const int MAX_MESSAGES      = 256; ///<@todo check this number being too high

//------------------------------------------------------------------------------------------------------------------------------
// Status values that we should use everywhere instead of magic numbers like 0, -1 etc
//------------------------------------------------------------------------------------------------------------------------------
enum Status
{
    S_ERROR = -1,
    S_OK    =  0,
};

//------------------------------------------------------------------------------------------------------------------------------
// Enum for different channels
//------------------------------------------------------------------------------------------------------------------------------
enum TFliteOutputs
{
    OUTPUT_ID_INVALID   = -1,
    OUTPUT_ID_RGB_IMAGE = 0,
    OUTPUT_ID_MAX_TYPES
};

static const int RgbOutputMask  = (1 << OUTPUT_ID_RGB_IMAGE);

#endif // COMMON_DEFS
