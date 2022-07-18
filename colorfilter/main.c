#include "bmp.h"

#include <stdio.h>
#include <stdlib.h>
#if __SSE2__
#include <x86intrin.h>
#endif

int is_apt_for_exercise(bmp_t *bmp)
{
	int is_argb =
		(bmp->channels == 4) &&
		(bmp->compression == BMP_BI_BITFIELDS) &&
		(bmp->redMask == BMP_ARGB8888_R_MASK) &&
		(bmp->greenMask == BMP_ARGB8888_G_MASK) &&
		(bmp->blueMask == BMP_ARGB8888_B_MASK) &&
		(bmp->alphaMask == BMP_ARGB8888_A_MASK);
	return is_argb;
}

#if __SSE2__

void filter1(uint8_t *in, uint8_t *out, int num_pixels)
{
	// pixel_t value;
	int p;

	//__m256i subVector = _mm256_set1_epi8(0xFF);
	for (p = 0; p < num_pixels*4 - 32; p += 32)
	{
		__m256i initial = _mm256_loadu_si256((__m256i *)&in[p]);
		__m256i current = _mm256_loadu_si256((__m256i *)&in[p]);

		//__m256i afterSubtraction = _mm256_subs_epu8(subVector, current);
		__m256i afterSubtraction = _mm256_xor_si256(current, _mm256_set1_epi32(-1));


		__m256i output = _mm256_blendv_epi8(initial, afterSubtraction, _mm256_set1_epi32(0x00FFFFFF));

		_mm256_storeu_si256((__m256i *)&out[p], output);
	}
	for (; p < num_pixels*4; p++)
	{
		out[p] = 255 - in[p];
	}

	FILE *fp;

	fp = fopen("test.txt", "w+");

	for (int i = 0; i < num_pixels; i += 4)
	{
		pixel_t *pixel = (pixel_t *)&out[i];
		pixel_t *pixelIn = (pixel_t *)&in[i];
		fprintf(fp, "%d:In: R:%d,G:%d,B:%d,A:%d || ", i / 4, pixelIn[i].r, pixelIn[i].g, pixelIn[i].b, pixelIn[i].a);
		fprintf(fp, "%d:Out R:%d,G:%d,B:%d,A:%d\n", i / 4, pixel[i].r, pixel[i].g, pixel[i].b, pixel[i].a);
	}

	fclose(fp);
}

#else

void filter1(uint8_t *in, uint8_t *out, int num_pixels)
{
	pixel_t value;
	int p;

	for (p = 0; p < num_pixels; p++)
	{

		value = ((pixel_t *)in)[p];
		value.r = 255 - value.r;
		value.g = 255 - value.g;
		value.b = 255 - value.b;
		((pixel_t *)out)[p] = value;
	}
}

#endif

void filter2(uint8_t *in, uint8_t *out, int num_pixels)
{
	pixel_t value;
	int p;

	for (p = 0; p < num_pixels; p++)
	{
		value = ((pixel_t *)in)[p];
		double greyValue = value.r * 0.2989 + value.g * 0.5870 + value.b * 0.1140;
		value.r = greyValue;
		value.g = greyValue;
		value.b = greyValue;
		((pixel_t *)out)[p] = value;
	}
}

void filter3(uint8_t *in, uint8_t *out, int num_pixels)
{
	pixel_t value;
	int p;

	for (p = 0; p < num_pixels; p++)
	{
		value = ((pixel_t *)in)[p];
		double greyValue = value.r * 0.2989 + value.g * 0.5870 + value.b * 0.1140;
		int isRedDominant = value.r > value.g && value.r > value.b;
		value.r = (isRedDominant) ? value.r : greyValue;
		value.g = (isRedDominant) ? value.g : greyValue;
		value.b = (isRedDominant) ? value.b : greyValue;
		((pixel_t *)out)[p] = value;
	}
}

void filter4(uint8_t *in, uint8_t *out, int num_pixels)
{
	pixel_t value;
	int p;

	for (p = 0; p < num_pixels; p++)
	{
		value = ((pixel_t *)in)[p];

		// Should be sepia filter, unfortunately weird blue spots appear
		double redValue = value.r * 0.393 + value.g * 0.769 + value.b * 0.189;
		double greenValue = value.r * 0.349 + value.g * 0.686 + value.b * 0.168;
		double blueValue = value.r * 0.272 + value.g * 0.534 + value.b * 0.131;

		value.r = redValue;
		value.g = greenValue;
		value.b = blueValue;
		((pixel_t *)out)[p] = value;
	}
}

/*----------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
	bmp_t bmp_in, bmp_out;
	if (argc < 2 || argc > 3)
	{
		fprintf(stderr, "Usage: %s [task = 1] bmp-file\n", argv[0]);
		exit(1);
	}

	char *filename = argv[1];
	char task = '1';
	if (argc == 3)
	{
		filename = argv[2];
		task = argv[1][0];
	}

	bmp_read(&bmp_in, filename);

	if (!is_apt_for_exercise(&bmp_in))
	{
		fprintf(stderr, "For the sake simplicity please provide a ARGB8888 image.\n");
		exit(4);
	}

	bmp_copyHeader(&bmp_out, &bmp_in);

	switch (task)
	{
	case '1':
		filter1(bmp_in.data, bmp_out.data, bmp_in.width * bmp_in.height);
		break;
	case '2':
		filter2(bmp_in.data, bmp_out.data, bmp_in.width * bmp_in.height);
		break;
	case '3':
		filter3(bmp_in.data, bmp_out.data, bmp_in.width * bmp_in.height);
		break;
	case '4':
		filter4(bmp_in.data, bmp_out.data, bmp_in.width * bmp_in.height);
		break;
	default:
		fprintf(stderr, "Invalid filter.\n");
		exit(5);
	}

	bmp_write(&bmp_out, "output.bmp");
	bmp_free(&bmp_in);
	bmp_free(&bmp_out);

	return 0;
}
