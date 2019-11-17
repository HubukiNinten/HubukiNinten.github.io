### Example: Augment된 이미지가 아닌 데이터 시각화하기



imgaug에는 바운딩 박스나 히트맵과 같은 이미지가 아닌 결과를 빠르게 시각화 할수 있는 많은 기능이 포함되어 있습니다.



import numpy as npimport imgaug as iaimage = np.zeros((64, 64, 3), dtype=np.uint8)# pointskps = [ia.Keypoint(x=10.5, y=20.5), ia.Keypoint(x=60.5, y=60.5)]kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)image\_with\_kps = kpsoi.draw\_on\_image(image, size=7, color=(0, 0, 255))ia.imshow(image\_with\_kps)# bbsbbsoi = ia.BoundingBoxesOnImage([   ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=30.5)], shape=image.shape)image\_with\_bbs = bbsoi.draw\_on\_image(image)image\_with\_bbs = ia.BoundingBox(    x1=50.5, y1=10.5, x2=100.5, y2=16.5).draw\_on\_image(image\_with\_bbs, color=(255, 0, 0), size=3)ia.imshow(image\_with\_bbs)# polygonspsoi = ia.PolygonsOnImage([   ia.Polygon([(10.5, 20.5), (50.5, 30.5), (10.5, 50.5)])], shape=image.shape)image\_with\_polys = psoi.draw\_on\_image(    image, alpha\_points=0, alpha\_face=0.5, color\_lines=(255, 0, 0))ia.imshow(image\_with\_polys)# heatmapshms = ia.HeatmapsOnImage(np.random.random(size=(32, 32, 1)).astype(np.float32),                         shape=image.shape)image\_with\_hms = hms.draw\_on\_image(image)ia.imshow(image\_with\_hms)

LineStrings 과 segmentation maps 도 위와 같은 방법을 지원합니다.

### Example: Augmenter 한 번만 사용하기

인터페이스는 기능 보강 인스턴스를 여러 번 재사용하도록 조정되어 있지만 한 번만 자유롭게 사용할 수도 있습니다. augmenter를 매번 인스턴스화하는 오버 헤드는 대개 무시할 만합니다.

from imgaug import augmenters as iaaimport numpy as npimages = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)# always horizontally flip each input imageimages\_aug = iaa.Fliplr(1.0)(images=images)# vertically flip each input image with 90% probabilityimages\_aug = iaa.Flipud(0.9)(images=images)# blur 50% of all images using a gaussian kernel with a sigma of 3.0images\_aug = iaa.Sometimes(0.5, iaa.GaussianBlur(3.0))(images=images)

### Example: 멀티코어 Augmentation

이미지는 augment\_batches(batches, background=True)방식을 이용하여 백그라운드 프로세스에서 보강될 수 있습니다. Batches는[imgaug.augmentables.batches.UnnormalizedBatch](https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmentables/batches.html#UnnormalizedBatch) or [imgaug.augmentables.batches.Batch](https://imgaug.readthedocs.io/en/latest/source/api_augmentables_batches.html#imgaug.augmentables.batches.Batch). 의 목록/생성기입니다. 아래의 예는 백그라운드에서 이미지 batch를 보강합니다.

import skimage.dataimport imgaug as iaimport imgaug.augmenters as iaafrom imgaug.augmentables.batches import UnnormalizedBatch# Number of batches and batch size for this examplenb\_batches =10batch\_size =32# Example augmentation sequence to run in the backgroundaugseq = iaa.Sequential([   iaa.Fliplr(0.5),    iaa.CoarseDropout(p=0.1, size\_percent=0.1)])# For simplicity, we use the same image here many timesastronaut = skimage.data.astronaut()astronaut = ia.imresize\_single\_image(astronaut, (64, 64))# Make batches out of the example image (here: 10 batches, each 32 times# the example image)batches = []for \_ inrange(nb\_batches):    batches.append(UnnormalizedBatch(images=[astronaut] \* batch\_size))# Show the augmented images.# Note that augment\_batches() returns a generator.for images\_aug in augseq.augment\_batches(batches, background=True):    ia.imshow(ia.draw\_grid(images\_aug.images\_aug, cols=8))

백그라운드 augmentation에 더 많은 통제가 필요하다면, (예: 시드 설정, 사용 된 CPU 코어 수 제어 또는 메모리 사용량 제한),그에 해당하는[multicore augmentation notebook](https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb) 과 [Augmenter.pool()](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Augmenter.pool) and [imgaug.multicore.Pool](https://imgaug.readthedocs.io/en/latest/source/api_multicore.html#imgaug.multicore.Pool). 에 대한 API를 참조하세요.

### Example: 매개변수로서의 확률 분포

대부분의 augmenter는 튜플 (a, b)을 균일 (a, b)을 나타내는 바로 가기로 사용하거나 목록 [a, b, c]를 사용하여 하나를 임의로 선택할 수있는 허용 된 값 세트를 나타냅니다. 더 복잡한 확률 분포 (예 : 가우시안, 잘린 가우시안 또는 포아송 분포)가 필요한 경우 imgaug.parameters에서 확률 매개 변수를 사용할 수 있습니다.

import numpy as npfrom imgaug import augmenters as iaafrom imgaug import parameters as iapimages = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)# Blur by a value sigma which is sampled from a uniform distribution# of range 10.1 \&lt;= x \&lt; 13.0.# The convenience shortcut for this is: GaussianBlur((10.1, 13.0))blurer = iaa.GaussianBlur(10+ iap.Uniform(0.1, 3.0))images\_aug = blurer(images=images)# Blur by a value sigma which is sampled from a gaussian distribution# N(1.0, 0.1), i.e. sample a value that is usually around 1.0.# Clip the resulting value so that it never gets below 0.1 or above 3.0.blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(1.0, 0.1), 0.1, 3.0))images\_aug = blurer(images=images)

라이브러리에는 더 많은 확률 분포가 있습니다 (예 : 절단 된 가우시안 분포, 포아송 분포 또는 베타 분포.

### Example: WithChannels

특정 이미지 채널에만 augmenter를 적용:

import numpy as npimport imgaug.augmenters as iaa# fake RGB imagesimages = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)# add a random value from the range (-30, 30) to the first two channels of# input images (e.g. to the R and G channels)aug = iaa.WithChannels(  channels=[0, 1],  children=iaa.Add((-30, 30)))images\_aug = aug(images=images)

### Example: Hooks



미리 정해진 순서에 따라 augmenter를 자유롭게 비활성화 할 수 있습니다. 여기서는 파이프 라인을 통해 두 번째 배열(히트맵)을 실행하여, 해당 입력에 augmenter의 부분 집합만 적용합니다.



import numpy as npimport imgaug as iaimport imgaug.augmenters as iaa# Images and heatmaps, just arrays filled with value 30.# We define the heatmaps here as uint8 arrays as we are going to feed them# through the pipeline similar to normal images. In that way, every# augmenter is applied to them.images = np.full((16, 128, 128, 3), 30, dtype=np.uint8)heatmaps = np.full((16, 128, 128, 21), 30, dtype=np.uint8)# add vertical lines to see the effect of flipimages[:, 16:128-16, 120:124, :] =120heatmaps[:, 16:128-16, 120:124, :] =120seq = iaa.Sequential([ iaa.Fliplr(0.5, name=&quot;Flipper&quot;),  iaa.GaussianBlur((0, 3.0), name=&quot;GaussianBlur&quot;),  iaa.Dropout(0.02, name=&quot;Dropout&quot;),  iaa.AdditiveGaussianNoise(scale=0.01\*255, name=&quot;MyLittleNoise&quot;),  iaa.AdditiveGaussianNoise(loc=32, scale=0.0001\*255, name=&quot;SomeOtherNoise&quot;),  iaa.Affine(translate\_px={&quot;x&quot;: (-40, 40)}, name=&quot;Affine&quot;)])# change the activated augmenters for heatmaps,# we only want to execute horizontal flip, affine transformation and one of# the gaussian noisesdefactivator\_heatmaps(images, augmenter, parents, default):    if augmenter.name in [&quot;GaussianBlur&quot;, &quot;Dropout&quot;, &quot;MyLittleNoise&quot;]:        returnFalse    else:        # default value for all other augmenters        return defaulthooks\_heatmaps = ia.HooksImages(activator=activator\_heatmaps)# call to\_deterministic() once per batch, NOT only once at the startseq\_det = seq.to\_deterministic()images\_aug = seq\_det(images=images)heatmaps\_aug = seq\_det(images=heatmaps, hooks=hooks\_heatmaps)

## Augmenter 리스트

다음은 사용 가능한 augmenter의 목록입니다. 아래에 언급 된 대부분의 변수는 범위로 설정할 수 있습니다 (예 : 이미지 당 0과 1.0 사이의 임의의 값을 샘플링하려면 A = (0.0, 1.0), 이미지 당 0.0이나 0.5 또는 1.0을 임의로 샘플링하려면 A = [0.0, 0.5, 1.0].

**산수**

| **Augmenter** | **정의** |
| --- | --- |
| Add(V, PCH) | V값을 각 이미지에 추가합니다. PCH가 참이라면, 샘플 값이 채널마다 달라집니다. |
| AddElementwise(V, PCH) | V값을 각 픽셀 단위에 첨가합니다. PCH가 참이라면, 샘플 값이 채널마다 달라집니다. (픽셀 마다) |
| AdditiveGaussianNoise(L, S, PCH) | 픽셀단위의 화이트 노이즈와 가우시안 노이즈를 이미지에 첨가합니다. 노이즈는 정규 분포 N(L,S) 를 따릅니다. PCH가 참이라면, 샘플 값이 채널마다 달라집니다. (픽셀 마다) |
| AdditiveLaplaceNoise(L, S, PCH) | Laplace (L, S)에 따라 laplace 분포에서 샘플링 된 노이즈를 이미지에 추가합니다. PCH가 true이면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있습니다. |
| AdditivePoissonNoise(L, PCH) | L이 람다 지수 인 포아송 분포에서 샘플링 된 노이즈를 추가합니다. PCH가 true이면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있습니다. |
| Multiply(V, PCH) | 각 이미지에 V 값을 곱하여 더 어둡고 밝은 이미지로 만듭니다. PCH가 참이면 샘플링 된 값이 채널마다 다를 수 있습니다. |
| MultiplyElementwise(V, PCH) | 각 픽셀에 값 V를 곱하여 더 어둡고 밝은 픽셀로 만듭니다. PCH가 true이면 샘플링 된 값이 채널 (및 픽셀)마다 다를 수 있습니다. |
| Dropout(P, PCH) | 확률이 P 인 픽셀을 0으로 설정합니다. PCH가 true이면 채널이 다르게 처리 될 수 있으며, 그렇지 않으면 전체 픽셀이 0으로 설정됩니다. |
| CoarseDropout(P, SPX, SPC, PCH) | 드롭 아웃과 유사하지만 픽셀 크기가 SPX이거나 상대적 크기가 SPC 인 거친 / 작은 이미지에서 0으로 설정 될 픽셀의 위치를 ​​샘플링합니다. 즉 SPC에 작은 값이 있으면 대략적인 맵이 작으므로 큰 사각형이 삭제됩니다. |
| ReplaceElementwise(M, R, PCH) | 이미지의 픽셀을 교체 R로 교체합니다. 마스크 M으로 식별 된 픽셀을 교체합니다. M은 예를 들어 모든 픽셀의 5 %를 대체하려면 0.05입니다. PCH가 참이면 마스크는 이미지, 픽셀 및 추가로 채널별로 샘플링됩니다. |
| ImpulseNoise(P) | 모든 픽셀의 P 퍼센트를 임펄스 노이즈, 즉 매우 밝거나 어두운 RGB 색상으로 대체합니다. SaltAndPepper (P, PCH = True)의 별칭입니다. |
| SaltAndPepper(P, PCH) | 모든 픽셀의 P 퍼센트를 매우 흰색 또는 검은 색으로 바꿉니다. PCH가 참이면 채널마다 다른 픽셀이 교체됩니다. |
| CoarseSaltAndPepper(P, SPX, SPC, PCH) | CoarseDropout과 유사하지만 영역을 0으로 설정하는 대신 매우 흰색 또는 검은 색으로 바뀝니다. PCH가 참이면, coarse 교체 마스크는 이미지 및 채널당 한 번 샘플링됩니다. |
| Salt(P, PCH) | SaltAndPepper와 유사하지만 검은 색이 아닌 매우 흰색으로 만 대체됩니다. |
| CoarseSalt(P, SPX, SPC, PCH) | CoarseSaltAndPepper와 유사하지만 검은 색이 아닌 매우 흰색으로 만 대체됩니다. |
| Pepper(P, PCH) | SaltAndPepper와 유사하지만 매우 검은 색으로 만 교체됩니다 (예 : 흰색 없음). |
| CoarsePepper(P, SPX, SPC, PCH) | CoarseSaltAndPepper와 유사하지만 매우 검은 색으로 만 교체됩니다. |
| Invert(P, PCH) | 이미지의 모든 픽셀을 확률 P로 반전합니다. 즉, (1-pixel\_value)로 설정합니다. PCH가 true이면 각 채널이 개별적으로 처리됩니다 (일부 채널 만 반전 됨). |
| ContrastNormalization(S, PCH) | 픽셀 값을 128보다 가까이 또는 더 가깝게 이동하여 이미지의 차이를 변경합니다. 방향과 강도는 S로 정의됩니다. PCH가 true로 설정되면 프로세스는 다른 가능한 S로 채널 단위로 발생합니다. |
| JpegCompression(C) | 강도 C (값 범위 : 0 ~ 100)의 JPEG 압축을 이미지에 적용합니다. C 값이 높을수록 시각적 인공물이 더 많이 나타납니다. |

**혼합**

| **Augmenter** | **정의** |
| --- | --- |
| Alpha(A, FG, BG, PCH) | Augmenter FG와 BG를 사용하여 이미지를 보강 한 다음 알파 A를 사용하여 결과를 혼합합니다. FG와 BG는 기본적으로 제공되지 않으면 아무 것도 수행하지 않습니다. 예 : 알파 (0.9, FG)를 사용하여 FG를 통해 이미지를 확대 한 다음 결과를 혼합하여 원래 이미지의 10 %를 유지합니다 (FG 이전). PCH가 true로 설정되면 프로세스는 A와 다르게 채널 단위로 발생합니다 (FG 및 BG는 이미지 당 한 번 계산 됨). |
| AlphaElementwise(A, FG, BG, PCH) | Alpha와 동일하지만 A에서 샘플링 된 연속 마스크 (값 0.0 ~ 1.0)를 사용하여 픽셀 단위로 블렌딩을 수행합니다. PCH가 true로 설정되면 프로세스는 픽셀 단위와 채널 단위로 발생합니다. |
| SimplexNoiseAlpha(FG, BG, PCH, SM, UP, I, AGG, SIG, SIGT) | Alpha와 유사하지만 마스크를 사용하여 augmenter FG 및 BG의 결과를 혼합합니다. 마스크는 단순 노이즈에서 샘플링되며, 이는 거친 경향이 있습니다. 마스크는 I 반복 (기본값 : 1 ~ 3)으로 수집되며 각 반복은 집계 방법 AGG (기본 최대, 즉 픽셀 당 모든 반복의 최대 값)를 사용하여 결합됩니다. 각 마스크는 최대 해상도 SM (기본값 2 ~ 16px)의 저해상도 공간에서 샘플링되며 UP 방법 (기본값 : 선형 또는 3 차 또는 가장 가까운 인접 업 샘플링)을 사용하여 이미지 크기로 업 스케일됩니다. SIG가 true이면 임계 값 SIGT를 사용하여 S 자형이 마스크에 적용되어 블롭의 값이 0.0 또는 1.0에 가까워집니다. |
| FrequencyNoiseAlpha(E, FG, BG, PCH, SM, UP, I, AGG, SIG, SIGT) | SimplexNoiseAlpha와 유사하지만 주파수 영역에서 노이즈 마스크를 생성합니다. 지수 E는 주파수 성분을 증가 / 감소시키는 데 사용됩니다. E의 값이 높으면 고주파 성분이 발음됩니다. 대략 -2에서 생성 된 구름 같은 패턴과 함께 -4에서 4 사이의 값을 사용하십시오. |

**블러**

| **Augmenter** | **정의** |
| --- | --- |
| GaussianBlur(S) | 크기가 S 인 가우스 커널을 사용하여 이미지를 흐리게합니다. |
| AverageBlur(K) | 크기가 K 인 간단한 평균 커널을 사용하여 이미지를 흐리게합니다. |
| MedianBlur(K) | K 크기의 중간 값을 통해 중앙값을 사용하여 이미지를 흐리게합니다. |
| BilateralBlur(D, SC, SS) | 거리 D (커널 크기 등)의 양방향 필터를 사용하여 이미지를 흐리게합니다. SC는 색 공간의 (영향) 거리에 대한 시그마이고, SS는 공간 거리에 대한 시그마입니다. |
| MotionBlur(K, A, D, O) | 크기가 K 인 모션 블러 커널을 사용하여 이미지를 흐리게합니다. A는 y 축에 대한 흐림 각도입니다 (값 범위 : 0-360, 시계 방향). D는 흐림 방향입니다 (값 범위 : -1.0 ~ 1.0, 1.0은 중앙에서 앞으로). O는 보간 순서입니다 (O = 0은 빠름, O = 1은 약간 느리지 만 더 정확합니다). |

**색상**

| **Augmenter** | **정의** |
| --- | --- |
| WithColorspace(T, F, CH) | 색상 공간 T에서 F로 이미지를 변환하고 자식 augmenter CH를 적용한 다음 F에서 T로 다시 변환합니다. |
| AddToHueAndSaturation(V, PCH, F, C) | HSV 공간의 각 픽셀에 값 V를 추가합니다 (예 : 색조 및 채도 수정). 색 공간 F에서 HSV로 변환합니다 (기본값은 F = RGB). 기능 보강하기 전에 채널 C를 선택합니다 (기본값은 C = [0,1]). PCH가 참이면 샘플링 된 값이 채널마다 다를 수 있습니다. |
| ChangeColorspace(T, F, A) | 색상 공간 F에서 T로 이미지를 변환하고 alpha A를 사용하여 원본 이미지와 혼합합니다. 회색조는 3 채널로 유지됩니다. (실제로 테스트되지 않은 augmenter 입니다, 위험은 본인이 감수 합시다.) |
| Grayscale(A, F) | 색상 공간 F (기본값 : RGB)에서 이미지를 회색조로 변환하고 alpha A를 사용하여 원본 이미지와 혼합합니다. |

**대조**

| **Augmenter** | **정의** |
| --- | --- |
| GammaContrast(G, PCH) | I\_ij &#39;= I\_ij \*\* G&#39;다음에 감마 대비 조정을 적용합니다. 여기서 G &#39;는 G에서 샘플링 된 감마 값이고 픽셀에서 I\_ij (0에서 1.0 공간으로 변환)입니다. PCH가 참이면 이미지와 채널마다 다른 G &#39;가 샘플링됩니다. |
| SigmoidContrast(G, C, PCH) | GammaContrast와 유사하지만 I\_ij &#39;= 1 / (1 + exp (G&#39;\* (C &#39;-I\_ij)))를 적용합니다. 여기서 G&#39;는 G에서 샘플링 된 이득 값이고 C &#39;는 C에서 샘플링 된 컷오프 값입니다. |
| LogContrast(G, PCH) | GammaContrast와 유사하지만 I\_ij = G &#39;\* log (1 + I\_ij)를 적용합니다. 여기서 G&#39;는 G에서 샘플링 된 이득 값입니다. |
| LinearContrast(S, PCH) | GammaContrast와 유사하지만 I\_ij = 128 + S &#39;\* (I\_ij-128)를 적용합니다. 여기서 S&#39;는 S에서 샘플링 된 강도 값입니다. 이 augmenter는 ContrastNormalization과 동일합니다 (향후 더 이상 사용되지 않음). |
| AllChannelsHistogramEqualization() | 각 입력 이미지의 각 채널에 표준 히스토그램 이퀄라이제이션을 적용합니다. |
| HistogramEqualization(F, T) | AllChannelsHistogramEqualization과 유사하지만 이미지가 색상 공간 F에있을 것으로 예상하고 색상 공간 T로 변환하고 강도 관련 채널 만 정규화합니다 (예 : T = Lab의 경우 L (T의 기본값) 또는 T = HSV의 V입니다. |
| AllChannelsCLAHE(CL, K, Kmin, PCH) | 클리핑 제한 CL 및 커널 사이즈 K (범위 [Kmin, inf)로 클리핑 됨)를 사용하여 각 이미지 채널에 적용되는 Limited Adaptive Histrogram Equalization을 대조합니다.(작은 이미지 패치의 Histogram Equalization). PCH가 참이면 채널마다 CL 및 K에 대한 다른 값이 샘플링됩니다. |
| CLAHE(CL, K, Kmin, F, T) | HistogramEqualization과 유사하게 Lab / HSV / HLS 색 공간의 강도 관련 채널에만 CLAHE를 적용합니다. (일반적으로 이것은 AllChannelsCLAHE보다 훨씬 잘 작동합니다.) |

**합성**

| **Augmenter** | **정의** |
| --- | --- |
| Convolve(M) | 람다 함수일 가능성이 있는 행렬 M으로 이미지를 통합합니다. |
| Sharpen(A, L) | 밝기 L로 각 이미지에 선명하게 커널을 실행합니다 (값이 낮 으면 이미지가 어두워집니다). Alpha A를 사용하여 결과를 원본 이미지와 혼합합니다. |
| Emboss(A, S) | 강도가 S 인 각 이미지에서 양각 커널을 실행합니다. alpha A를 사용하여 결과를 원본 이미지와 혼합합니다. |
| EdgeDetect(A) | 각 이미지에서 엣지 검출커널을 실행합니다. Alpha A를 사용하여 결과를 원본 이미지와 혼합합니다. |
| DirectedEdgeDetect(A, D) | 각 이미지에 대해 방향 지정 에지 감지 커널을 실행하여 각 방향 D에서 감지합니다 (기본값 : 이미지 당 선택한 0에서 360 도의 임의 방향). alpha A를 사용하여 결과를 원본 이미지와 혼합합니다. |

**엣지**

| **Augmenter** | **정의** |
| --- | --- |
| Canny(A, HT, SK, C) | 히스테리시스 임계 값 HT 및 소벨 커널 크기 SK를 사용하여 각 이미지에 캐니 엣지 감지를 적용합니다. 클래스 C를 사용하여 이진 이미지를 색상으로 변환합니다. 알파는 요소 A를 사용하여 입력 이미지와 혼합됩니다. |

**뒤집기**

| **Augmenter** | **정의** |
| --- | --- |
| Fliplr(P) | 확률 P로 이미지를 가로로 뒤집습니다. |
| Flipud(P) | 확률 P로 이미지를 세로로 뒤집습니다. |

**기하**

| **Augmenter** | **정의** |
| --- | --- |
| Affine(S, TPX, TPC, R, SH, O, CVAL, FO, M, B) | 이미지에 아핀 변환을 적용합니다. S로 스케일을 조정하고 (\&gt; 1 = 확대, \&lt;1 = 확대), TPX 픽셀 또는 TPC 백분율로 변환하고, R 도씩 회전하고 SH도만큼 기울입니다. 순서 O로 보간이 발생합니다 (0 또는 1이 양호하고 빠름). FO가 참이면 출력 이미지 평면 크기가 왜곡 된 이미지 크기에 맞춰집니다. 즉 45도 회전 한 이미지는 이미지 평면 외부에 있지 않습니다. M은 입력 이미지 평면에 해당하지 않는 출력 이미지 평면의 픽셀을 처리하는 방법을 제어합니다. M = &#39;constant&#39;이면 CVAL은 이러한 픽셀을 채울 상수 값을 정의합니다. B는 백엔드 프레임 워크 (현재 cv2 또는 skimage)를 설정할 수 있습니다. |
| AffineCv2(S, TPX, TPC, R, SH, O, CVAL, M, B) | Affine과 동일하지만 백엔드로 cv2 만 사용합니다. 현재 FO = true를 지원하지 않습니다. 향후에는 더 이상 사용되지 않을 수 있습니다. |
| PiecewiseAffine(S, R, C, O, M, CVAL) | 이미지에 일정한 점 격자를 배치합니다. 그리드에는 R 행과 C 열이 있습니다. 그런 다음 정규 분포 N (0, S)의 샘플 인 양만큼 점 (및 그 주변의 이미지 영역)을 이동하여 다양한 강도의 로컬 왜곡을 일으 킵니다. O, M 및 CVAL은 Affine에서와 같이 정의됩니다. |
| PerspectiveTransform(S, KS) | 임의의 4 점 투시 변환을 이미지에 적용합니다 (고급 자르기 형태와 유사). 각 점은 시그마 S를 사용한 정규 분포에서 파생 된 이미지 코너로부터 임의의 거리를 갖습니다. KS가 True (기본값)로 설정되면 각 이미지의 크기가 원래 크기로 다시 조정됩니다. |
| ElasticTransformation(S, SM, O, CVAL, M) | 왜곡 필드를 기준으로 각 픽셀을 개별적으로 이동합니다. SM은 왜곡 필드의 평활도와 S 강도를 정의합니다. O는 보간 순서이며, CVAL은 새로 생성 된 픽셀에 대한 상수 채우기 값이고 M은 채우기 모드입니다 (증강 기 Affine 참조). |
| Rot90(K, KS) | 이미지를 시계 방향으로 90도 회전합니다. (이것은 Affine보다 빠릅니다.) KS가 true이면 결과 이미지는 원래 입력 이미지와 동일한 크기로 크기가 조정됩니다. |

**메타**

| **Augmenter** | **정의** |
| --- | --- |
| Sequential(C, R) | 자식 augmenter C의 목록을 가져 와서 이 순서대로 이미지에 적용합니다. R이 true이면 (기본값 : false) 순서는 무작위입니다 (배치 당 한 번 선택). |
| SomeOf(N, C, R) | Augementer C 목록에서 임의로 선택된 N 개의 augmenter를 각 이미지에 적용합니다. Augmenter는 이미지마다 선택됩니다. R은 Sequential과 동일합니다. N은 범위 일 수 있습니다 (예 : 1에서 3을 선택하기 위해 (1, 3). |
| OneOf(C) | SomeOf(1, C)와 동일. |
| Sometimes(P, C, D) | 자식 augmenter C를 사용하여 확률 P로 이미지를 보강하고, 그렇지 않으면 D를 사용합니다. D는 없음 일 수 있으며, 모든 이미지의 P % 만 C를 통해 증강. |
| WithColorspace(T, F, C) | 이미지를 색상 공간 F (기본값 : RGB)에서 색상 공간 T로 변환하고, augmenter C를 적용한 다음 다시 F로 변환합니다. |
| WithChannels(H, C) | 각 이미지 채널 H (예 : RGB 이미지에서 빨강 및 녹색의 경우 [0,1])에서 선택하고 자식 augmenter C를 이 채널에 적용하고 결과를 원래 이미지로 다시 병합합니다. |
| Noop() | 아무것도하지 않습니다. (검증 / 테스트에 유용합니다.) |
| Lambda(I, K) | 람다 함수 I을 이미지에 적용하고 K를 키포인트에 적용합니다. |
| AssertLambda(I, K) | 람다 함수 I을 통해 이미지를 확인하고 K를 통해 키포인트를 확인하고 둘 중 하나에 의해 false가 반환되면 오류가 발생합니다. |
| AssertShape(S) | 입력 이미지의 모양이 S가 아닌 경우 오류가 발생합니다. |
| ChannelShuffle(P, C) | 모든 이미지의 P 퍼센트에 대한 색상 채널 순서를 변경합니다. 기본적으로 모든 채널을 셔플하지만 C (채널 인덱스 목록)를 사용하는 하위 세트로 제한 할 수 있습니다. |

**풀링**

| **Augmenter** | **정의** |
| --- | --- |
| AveragePooling(K, KS) | 커널 크기가 K 인 평균 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하십시오. |
| MaxPooling(K, KS) | 커널 크기가 K 인 최대 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하십시오. |
| MinPooling(K, KS) | 커널 크기가 K 인 최소 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하십시오. |
| MedianPooling(K, KS) | 커널 크기가 K 인 중앙 풀. KS = True이면 풀링 된 이미지의 크기를 입력 이미지 크기로 다시 조정하십시오. |

**분할**

| **Augmenter** | **정의** |
| --- | --- |
| Superpixels(P, N, M) | (최대) 해상도 M에서 이미지의 N 수퍼 픽셀을 생성하고 원래 크기로 다시 크기를 조정합니다. 그런 다음 원본 이미지의 모든 수퍼 픽셀 영역의 P %가 수퍼 픽셀로 대체됩니다. (1-P) 퍼센트는 변경되지 않습니다. |
| Voronoi(PS, P, M) | Voronoi 셀의 좌표를 얻기 위해 샘플러 PS를 쿼리합니다. 각 셀에서 모든 픽셀을 프로브로 바꿉니다. 평균으로 P. 최대 해상도 M에서이 단계를 수행합니다. |
| UniformVoronoi(N, P, M) | 각 이미지에 N Voronoi 셀을 무작위로 배치합니다. 각 셀에서 모든 픽셀을 프로브로 바꿉니다. 평균으로 P. 최대 해상도 M에서이 단계를 수행합니다. |
| RegularGridVoronoi(H, W, P, M) | 각 이미지에 N Voronoi 셀을 무작위로 배치합니다. 각 셀에서 모든 픽셀을 프로브로 바꿉니다. 평균으로 P. 최대 해상도 M에서이 단계를 수행합니다. |
| RelativeRegularGridVoronoi(HPC, WPC, P, M) | 각 이미지에 HPC \* H x WPC \* W Voronoi 셀의 규칙적인 그리드를 배치합니다 (H, W는 이미지 높이 및 너비). 각 셀에서 모든 픽셀을 프로브로 바꿉니다. 평균으로 P. 최대 해상도 M에서이 단계를 수행합니다. |

**크기**

| **Augmenter** | **정의** |
| --- | --- |
| Resize(S, I) | 이미지의 크기를 S로 조정합니다. 일반적인 사용 사례는 S = { &quot;height&quot;: H, &quot;width&quot;: W}를 사용하여 모든 이미지의 크기를 HxW 모양으로 조정하는 것입니다. H와 W는 플로트 일 수 있습니다 (예 : 원래 크기의 50 %로 크기 조정). H 또는 W는 한쪽의 새 크기 만 정의하고 다른 쪽의 크기를 적절하게 조정하기 위해 &quot;종횡비 유지&quot;일 수 있습니다. I는 (기본값 : 입방)을 사용하기 위한 보간입니다. |
| CropAndPad(PX, PC, PM, PCV, KS) | 이미지의 위 / 오른쪽 / 아래 / 왼쪽에서 PX 픽셀 또는 픽셀의 PC 백분율을 자르거나 채 웁니다. 음수 값은 자르기, 패딩 양수를 초래합니다. PM은 패드 모드를 정의합니다 (예 : 추가 된 모든 픽셀에 균일 한 색상 사용). PMV가 일정한 경우 PCV는 추가 된 픽셀의 색상을 제어합니다. KS가 true (기본값)이면 결과 이미지가 원래 크기로 다시 크기 조정됩니다. |
| Pad(PX, PC, PM, PCV, KS) | CropAndPad () 바로 가기는 픽셀 만 추가합니다. PX 및 PC에는 양수 값만 허용됩니다. |
| Crop(PX, PC, KS) | 픽셀 만 잘라내는 CropAndPad ()의 숏컷입니다. PX 및 PC에는 양수 값만 사용할 수 있습니다 (예 : 5 값은 5 픽셀이 잘립니다). |
| PadToFixedSize(W, H, PM, PCV, POS) | 높이 H와 너비 W까지의 모든 이미지를 채 웁니다. PM 및 PCV는 패드와 동일합니다. POS는 예를 들어 패딩 할 위치를 정의합니다. POS = &quot;center&quot;는 모든면에 똑같이, POS = &quot;left-top&quot;은 윗면과 왼쪽 만 채 웁니다. |
| CropToFixedSize(W, H, POS) | PadToFixedSize와 비슷하지만 패딩 대신 높이 H와 너비 W로 자릅니다. |
| KeepSizeByResize(CH, I, IH) | 자식 augmenter CH (예 : 자르기)를 적용한 후 모든 이미지의 크기를 원래 크기로 다시 조정합니다. I는 이미지에 사용 된 보간이고, IH는 히트 맵에 사용되는 보간입니다. |

**날씨**

| **Augmenter** | **정의** |
| --- | --- |
| FastSnowyLandscape(LT, LM) | HLS 색상 공간에서 L \&lt;LT를 갖는 모든 픽셀의 밝기 L을 LM의 계수로 증가시켜 풍경 이미지를 눈 풍경으로 변환 |
| Clouds() | 다양한 모양과 밀도의 구름을 이미지에 추가합니다. 오버레이 augmenter와 결합하는 것이 좋습니다.(예:SimplexNoiseAlpha.) |
| Fog() | 다양한 모양과 밀도의 안개 같은 구름 구조를 이미지에 추가합니다. 오버레이 augmenter와 결합하는 것이 좋습니다. (예:SimplexNoiseAlpha.) |
| CloudLayer(IM, IFE, ICS, AMIN, AMUL, ASPXM, AFE, S, DMUL) | 이미지에 단일 구름 레이어를 추가합니다. IM은 구름의 평균 강도, IFE는 강도에 대한 주파수 노이즈 지수 (고르지 않은 색상으로 이어짐), ICS는 강도 샘플링을 위한 가우시안의 분산을 제어하고 AM은 구름의 최소 불투명도 (값\&gt; 0은 일반적인 안개), 불투명도 값의 승수 AMUL, ASPXM은 불투명도 값을 샘플링 할 최소 그리드 크기를 제어하고, AFE는 불투명도 값의 주파수 노이즈 지수, S는 구름의 희소성을 제어하고 DMUL은 구름 밀도 멀티 플라이어입니다. 이 인터페이스는 최종적이 아니며 향후 변경 될 수 있습니다. |
| Snowflakes(D, DU, FS, FSU, A, S) | 밀도 D, 밀도 균일도 DU, 눈송이 크기 FS, 눈송이 크기 균일도 FSU, 떨어지는 각도 A 및 속도 S를 가진 눈송이를 이미지에 추가합니다. 1 ~ 3 층의 눈송이가 추가되므로 값은 확률 적이어야합니다. |
| SnowflakesLayer(D, DU, FS, FSU, A, S, BSF, BSL) | 이미지에 눈송이의 단일 레이어를 추가합니다. Snowflakes augmenter를 참조하십시오. BSF 및 BSL은 눈송이에 적용되는 가우시안 블러를 제어합니다. |