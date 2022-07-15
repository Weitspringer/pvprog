#include "bmp.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*----------------------------------------------------------------------------*/

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

static void swap_endianness_16bit(uint16_t *value)
{
	*value = *value << 8 | *value >> 8;
}

static void swap_endianness_32bit(uint32_t *value)
{
	*value = *value >> 24 | (*value >> 8 & 0xFF00) | ((*value & 0xFF00) << 8) | *value << 24;
}

static void fix_header(bmp_t *bmp)
{
	swap_endianness_32bit(&bmp->bytes);
	swap_endianness_32bit(&bmp->reserved);
	swap_endianness_32bit(&bmp->offset);
	swap_endianness_32bit(&bmp->infoHeaderSize);
	swap_endianness_32bit((uint32_t*)&bmp->width);
	swap_endianness_32bit((uint32_t*)&bmp->height);
	swap_endianness_16bit(&bmp->planes);
	swap_endianness_16bit(&bmp->bitCount);
	swap_endianness_32bit(&bmp->compression);
	swap_endianness_32bit(&bmp->sizeImage);
	swap_endianness_32bit((uint32_t*)&bmp->xPPM);
	swap_endianness_32bit((uint32_t*)&bmp->yPPM);
	swap_endianness_32bit(&bmp->clrUsed);
	swap_endianness_32bit(&bmp->clrImportant);
	swap_endianness_32bit(&bmp->redMask);
	swap_endianness_32bit(&bmp->greenMask);
	swap_endianness_32bit(&bmp->blueMask);
	swap_endianness_32bit(&bmp->alphaMask);
	swap_endianness_32bit(&bmp->csType);
	swap_endianness_32bit(&bmp->gammaRed);
	swap_endianness_32bit(&bmp->gammaGreen);
	swap_endianness_32bit(&bmp->gammaBlue);
	swap_endianness_32bit(&bmp->intent);
	swap_endianness_32bit(&bmp->profileData);
	swap_endianness_32bit(&bmp->profileSize);
	swap_endianness_32bit(&bmp->reserved_v5);
}

#else

#define fix_header(bmp)

#endif

/*----------------------------------------------------------------------------*/

void bmp_read(bmp_t *bmp, const char *filename)
{
	FILE *f = fopen(filename, "r");

	if (!f) {
		printf("Cannot open %s for reading.\n", filename);
		exit(2);
	}

	fread(bmp->header, sizeof(bmp->header), 1, f);
	fix_header(bmp);

	if (!(bmp->header[0] == 'B' && bmp->header[1] == 'M' && bmp->infoHeaderSize == BMP_DIB5_HEADER_SIZE)) {
		fprintf(stderr, "Unsupported bitmap format.\n");
		exit(3);
	}

	/* We are storing 4 channel bmp in the end. Override old value */
	bmp->channels = bmp->bitCount / 8;
	bmp->bitCount = 32;

	/* top-down bitmaps (top row first) are stored with negative height */
	bmp->isTopDown = bmp->height < 0;
	bmp->height = abs(bmp->height);

	/* To calculate right row in GET_PIXEL/SET_PIXEL */
	bmp->yDirection = bmp->isTopDown ? 1 : -1;
	bmp->yOffset = bmp->isTopDown ? 0 : bmp->height - 1;

	/* Allocate memory for our pixel data and read file */
	bmp->sizeImage = bmp->width * bmp->height * bmp->channels;
	bmp->data = malloc(bmp->sizeImage);
	fread(bmp->data, bmp->sizeImage, 1, f);

	fclose(f);
}

void bmp_copyHeader(bmp_t *bmp, bmp_t *other)
{
	memcpy(bmp, other, sizeof(bmp_t));
	bmp->data = calloc(bmp->sizeImage, 1);
}

void bmp_write(bmp_t *bmp, const char *filename)
{
	int x, y;
	pixel_t output;
	FILE *f = fopen(filename, "w");

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	bmp_t header_le;
	memcpy(header_le.header, bmp->header, sizeof(header_le.header));

	fix_header(&header_le);

	fwrite(header_le.header, sizeof(header_le.header), 1, f);
#else
	fwrite(bmp->header, sizeof(bmp->header), 1, f);
#endif

	/* Store bitmap bottom-up (thus positive height) */
	for (y = bmp->height - 1; y >= 0; y--) {
		for (x = 0; x < bmp->width; x++) {
			output = ((pixel_t *)bmp->data)[(y * bmp->yDirection + bmp->yOffset) * bmp->width + x];
			if (fwrite(&output, sizeof(output), 1, f) != 1) {
				fprintf(stderr, "Cannot write to file %s.\n", filename);
				exit(6);
			}
		}
	}

	fclose(f);
}

void bmp_free(bmp_t *bmp)
{
	free(bmp->data);
	bmp->data = NULL;
}

