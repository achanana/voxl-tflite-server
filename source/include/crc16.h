/*
 * Copyright (c) 2015-2016 Qualcomm Technologies, Inc.  All Rights Reserved.
 * Qualcomm Technologies Proprietary and Confidential.
 */
#ifndef CRC16_H_
#define CRC16_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

uint16_t crc16_init();

uint16_t crc16_byte(uint16_t crc, const uint8_t c);

uint16_t crc16(uint16_t crc, uint8_t const *buffer, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif //CRC16_H_
