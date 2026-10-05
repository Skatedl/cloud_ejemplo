# Guía 3 – Segmentación automática con SAM (MASATI-v2)

Curso: **Visión por Computador e IA** · Parcial 1 · Teledetección

Esta guía continúa tu código actual (`parse_voc_xml`, `SCENES`, `DATA_ROOT`) y cubre los 8 puntos del reto. Recomendado ejecutarla en **Google Colab con GPU** (Runtime → T4).

---

## 0. ¿Qué es SAM? (punto 1)

**Segment Anything Model (SAM)** es un modelo de segmentación de imágenes propuesto por Meta AI (Kirillov et al., 2023) y entrenado con el dataset **SA-1B** (más de 1.100 millones de máscaras sobre ~11 millones de imágenes). Es un *foundation model* para segmentación **promptable**: no está entrenado para una clase concreta, sino para devolver una máscara válida dado un prompt.

Tiene tres componentes:

| Componente | Función |
|---|---|
| **Image encoder** (ViT-H / ViT-L / ViT-B, preentrenado con MAE) | Calcula una sola vez el *embedding* de la imagen (la parte costosa). |
| **Prompt encoder** | Codifica prompts: puntos (positivos/negativos), **cajas**, máscaras previas o texto (en el paper, de forma exploratoria). |
| **Mask decoder** (ligero) | Combina embedding + prompt y produce máscaras (hasta 3 candidatas con su puntaje de IoU estimado) en milisegundos. |

Dos modos de uso que emplearemos:

- **`SamAutomaticMaskGenerator`**: sin prompts, cuadrícula de puntos sobre la imagen → segmenta "todo" (puntos 3 y 4 de la guía).
- **`SamPredictor`**: segmentación guiada por prompt, aquí **bounding box** (puntos 5 y 6).

Limitaciones relevantes en teledetección: SAM se entrenó con imágenes naturales, no sensa remotos; objetos pequeños (barcos de pocos píxeles), estelas, sombras y reflejos del agua pueden degradar la máscara.

---

## 1. Instalación y carga del modelo

```python
!pip install -q git+https://github.com/facebookresearch/segment-anything.git opencv-python
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Alternativa más liviana (menos VRAM): sam_vit_b_01ec64.pth
```

```python
import torch, cv2, shutil
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'vit_h'                      # 'vit_b' si falta memoria
CHECKPOINT = 'sam_vit_h_4b8939.pth'

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(device)
predictor = SamPredictor(sam)
print('SAM cargado en', device)
```

---

## 2. Selección de la imagen con ≥ 20 barcos (punto 2)

Se recorren todas las carpetas con etiquetas y se cuentan los barcos por XML.

```python
def resolve_image_path(img_dir, xml_path):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.bmp'):
        p = os.path.join(DATA_ROOT, img_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'No se halló imagen para {xml_path}')

rows = []
for scene, (img_dir, lbl_dir) in SCENES.items():
    if lbl_dir is None:
        continue
    for xml in glob.glob(os.path.join(DATA_ROOT, lbl_dir, '*.xml')):
        fn, boxes, names = parse_voc_xml(xml)
        rows.append(dict(scene=scene, xml=xml, n_ships=len(boxes)))

df = pd.DataFrame(rows).sort_values('n_ships', ascending=False)
print(df.head(10))
print('Imágenes con >= 20 barcos:', (df.n_ships >= 20).sum())
```

```python
MIN_SHIPS = 20
cand = df[df.n_ships >= MIN_SHIPS]
assert len(cand) > 0, 'Ninguna imagen llega a 20 barcos: revisa DATA_ROOT o baja MIN_SHIPS.'

# Elegimos la de más barcos (o fija una manualmente con cand.iloc[k])
sel = cand.iloc[0]
img_dir, lbl_dir = SCENES[sel.scene]
XML_PATH = sel.xml
IMG_PATH = resolve_image_path(img_dir, XML_PATH)

_, BOXES, NAMES = parse_voc_xml(XML_PATH)
image = np.array(Image.open(IMG_PATH).convert('RGB'))
print(f'Imagen: {IMG_PATH} | tamaño: {image.shape} | barcos: {len(BOXES)}')

def draw_boxes(ax, boxes, color='lime', lw=1.5):
    for (x0, y0, x1, y1) in boxes:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=color, linewidth=lw))

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image); draw_boxes(ax, BOXES)
ax.set_title(f'{len(BOXES)} barcos anotados (cajas originales)'); ax.axis('off')
plt.show()
```

---

## 3. Recorte de un grupo de barcos y SAM automático (punto 3)

Se elige un barco "semilla" y sus vecinos más cercanos; el recorte es la unión de sus cajas más un margen.

```python
def crop_around_ships(image, boxes, k=6, margin=40):
    centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])
    # semilla: el barco con vecinos más cercanos en promedio
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    seed = np.argmin(np.sort(d, axis=1)[:, :k].mean(axis=1))
    idx = np.argsort(d[seed])[:k]
    sub = [boxes[i] for i in idx]

    H, W = image.shape[:2]
    x0 = int(max(min(b[0] for b in sub) - margin, 0))
    y0 = int(max(min(b[1] for b in sub) - margin, 0))
    x1 = int(min(max(b[2] for b in sub) + margin, W))
    y1 = int(min(max(b[3] for b in sub) + margin, H))
    crop = image[y0:y1, x0:x1]
    # cajas de TODOS los barcos que caen dentro del recorte, en coords locales
    local = [(b[0]-x0, b[1]-y0, b[2]-x0, b[3]-y0) for b in boxes
             if b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1]
    return crop, local, (x0, y0, x1, y1)

crop, crop_boxes, crop_xyxy = crop_around_ships(image, BOXES)
print('Recorte (x0,y0,x1,y1):', crop_xyxy, '| barcos dentro:', len(crop_boxes))

mask_gen = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=20,
)
auto_masks = mask_gen.generate(crop)
print('Máscaras automáticas generadas:', len(auto_masks))

def show_anns(anns, ax):
    anns = sorted(anns, key=lambda a: a['area'], reverse=True)
    overlay = np.zeros((*anns[0]['segmentation'].shape, 4))
    overlay[..., 3] = 0
    for a in anns:
        m = a['segmentation']
        overlay[m] = np.concatenate([np.random.random(3), [0.55]])
    ax.imshow(overlay)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(crop); axs[0].set_title('Recorte')
axs[1].imshow(crop); draw_boxes(axs[1], crop_boxes); axs[1].set_title('Cajas originales')
axs[2].imshow(crop); show_anns(auto_masks, axs[2]); axs[2].set_title(f'SAM automático ({len(auto_masks)} máscaras)')
for a in axs: a.axis('off')
plt.tight_layout(); plt.show()
```

**Qué observar y anotar para el análisis (punto 4):**

- ¿SAM segmenta cada barco como un objeto separado o lo fusiona con su estela/sombra?
- ¿Cuántas máscaras corresponden a barcos y cuántas a mar, olas, nubes o ruido? (SAM automático **no sabe qué es un barco**: segmenta "todo").
- ¿Se pierden barcos pequeños o de bajo contraste?
- ¿Hay sobre-segmentación (un barco partido en cubierta, casco, proa)?

Tabla opcional para cuantificar el modo automático:

```python
def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1-ix0, 0) * max(iy1-iy0, 0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

auto_xyxy = [(m['bbox'][0], m['bbox'][1], m['bbox'][0]+m['bbox'][2], m['bbox'][1]+m['bbox'][3])
             for m in auto_masks]

recall_rows = []
for i, gb in enumerate(crop_boxes):
    best = max((box_iou(gb, ab) for ab in auto_xyxy), default=0)
    recall_rows.append(dict(barco=i, mejor_IoU_caja=round(best, 3), detectado=best >= 0.5))
rec = pd.DataFrame(recall_rows)
print(rec)
print(f'Barcos del recorte recuperados por SAM automático (IoU>=0.5): {rec.detectado.mean():.0%}')
```

---

## 4. SAM con caja como prompt sobre la imagen completa (puntos 5 y 6)

El *embedding* de la imagen se calcula **una sola vez** con `set_image`; luego cada barco es un prompt barato.

```python
def mask_to_polygon(mask, eps_ratio=0.01):
    """Máscara binaria -> polígono (N,2) del contorno externo más grande."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = eps_ratio * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)
    return c.reshape(-1, 2)

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def segment_ships_with_boxes(image, boxes, predictor, multimask=False, pad=0):
    """Ejecuta SAM individualmente por cada caja. Devuelve lista de dicts."""
    predictor.set_image(image)               # embedding una sola vez
    H, W = image.shape[:2]
    results = []
    for i, b in enumerate(boxes):
        prompt = np.array([max(b[0]-pad, 0), max(b[1]-pad, 0),
                           min(b[2]+pad, W), min(b[3]+pad, H)], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=prompt, multimask_output=multimask)
        k = int(np.argmax(scores))           # con multimask=False solo hay una
        m = masks[k]
        results.append(dict(idx=i, mask=m, score=float(scores[k]),
                            bbox=mask_to_bbox(m), poly=mask_to_polygon(m),
                            area=int(m.sum())))
    return results

sam_results = segment_ships_with_boxes(image, BOXES, predictor)
print('Barcos segmentados:', len(sam_results))
```

### Visualización

```python
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(image); draw_boxes(axs[0], BOXES, 'lime')
axs[0].set_title('Cajas originales (XML)')

axs[1].imshow(image)
rng = np.random.default_rng(0)
for r in sam_results:
    ov = np.zeros((*r['mask'].shape, 4))
    ov[r['mask']] = np.concatenate([rng.random(3), [0.6]])
    axs[1].imshow(ov)
    if r['bbox']:
        draw_boxes(axs[1], [r['bbox']], 'red', 1)
axs[1].set_title('Máscaras SAM + caja ajustada (rojo)')
for a in axs: a.axis('off')
plt.tight_layout(); plt.show()

# Zoom de 8 barcos: original vs SAM
sel_ids = list(range(min(8, len(BOXES))))
fig, axs = plt.subplots(2, len(sel_ids), figsize=(3*len(sel_ids), 6))
for j, i in enumerate(sel_ids):
    x0, y0, x1, y1 = [int(v) for v in BOXES[i]]
    p = 10
    ys, xs = slice(max(y0-p, 0), y1+p), slice(max(x0-p, 0), x1+p)
    axs[0, j].imshow(image[ys, xs]); axs[0, j].set_title(f'#{i}'); axs[0, j].axis('off')
    axs[1, j].imshow(image[ys, xs])
    axs[1, j].imshow(np.ma.masked_where(~sam_results[i]['mask'][ys, xs],
                                        sam_results[i]['mask'][ys, xs]), alpha=0.6, cmap='autumn')
    axs[1, j].axis('off')
axs[0, 0].set_ylabel('Original'); axs[1, 0].set_ylabel('SAM')
plt.tight_layout(); plt.show()
```

### Métricas de ajuste

```python
metr = []
for r, b in zip(sam_results, BOXES):
    box_area = (b[2]-b[0]) * (b[3]-b[1])
    metr.append(dict(
        barco=r['idx'],
        score_SAM=round(r['score'], 3),
        area_caja=round(box_area, 1),
        area_mascara=r['area'],
        relleno=round(r['area'] / box_area, 3),                  # fracción de la caja ocupada por el barco
        IoU_caja_vs_bbox_mascara=round(box_iou(b, r['bbox']), 3) if r['bbox'] else 0,
    ))
metr = pd.DataFrame(metr)
display(metr)
print(metr[['score_SAM', 'relleno', 'IoU_caja_vs_bbox_mascara']].describe().round(3))
```

Interpretación sugerida:

- **`relleno` bajo** (p. ej. < 0.5): la caja original era holgada o el barco es diagonal; la máscara ajusta mucho.
- **`relleno` ≈ 1** o **`score_SAM` bajo**: posible fallo (SAM tomó toda la caja) o barco muy pequeño/ambiguo.
- **`IoU_caja_vs_bbox_mascara` bajo**: la máscara se salió o se encogió respecto a la anotación; revisarlo visualmente.

---

## 5. Nuevo dataset `MASATI-v2_SAM` con XML actualizados (punto 7)

Se copian los XML a una nueva ubicación con sufijo `_SAM`. En cada `<object>`:

- `<bndbox>` se **actualiza** con la caja ajustada a la máscara.
- Se conserva la caja original en `<bndbox_original>` (trazabilidad).
- Se agrega `<segmentation>` con el polígono, el área y el score de SAM.

```python
DATA_ROOT_SAM = DATA_ROOT + '_SAM'

def write_sam_xml(xml_in, xml_out, results):
    tree = ET.parse(xml_in)
    root = tree.getroot()
    objs = root.findall('object')
    assert len(objs) == len(results), 'Cantidad de objetos y resultados no coincide'

    for obj, r in zip(objs, results):
        bb = obj.find('bndbox')

        # 1) guardar caja original
        orig = ET.SubElement(obj, 'bndbox_original')
        for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
            ET.SubElement(orig, tag).text = bb.find(tag).text

        # 2) actualizar caja (si SAM falló, se conserva la original)
        if r['bbox'] is not None:
            x0, y0, x1, y1 = r['bbox']
            bb.find('xmin').text, bb.find('ymin').text = str(x0), str(y0)
            bb.find('xmax').text, bb.find('ymax').text = str(x1), str(y1)

        # 3) segmentación como polígono
        seg = ET.SubElement(obj, 'segmentation')
        ET.SubElement(seg, 'source').text = f'SAM_{MODEL_TYPE}'
        ET.SubElement(seg, 'score').text = f"{r['score']:.4f}"
        ET.SubElement(seg, 'area').text = str(r['area'])
        poly = r['poly']
        ET.SubElement(seg, 'polygon').text = (
            ';'.join(f'{int(x)},{int(y)}' for x, y in poly) if poly is not None else ''
        )

    ET.indent(tree, space='  ')
    os.makedirs(os.path.dirname(xml_out), exist_ok=True)
    tree.write(xml_out, encoding='utf-8', xml_declaration=True)

# Rutas espejo: MASATI-v2/<scene>_labels/x.xml  ->  MASATI-v2_SAM/<scene>_labels/x.xml
xml_out = XML_PATH.replace(DATA_ROOT, DATA_ROOT_SAM, 1)
write_sam_xml(XML_PATH, xml_out, sam_results)

# Copiar también la imagen para que el nuevo dataset quede autocontenido
img_out = IMG_PATH.replace(DATA_ROOT, DATA_ROOT_SAM, 1)
os.makedirs(os.path.dirname(img_out), exist_ok=True)
shutil.copy2(IMG_PATH, img_out)

print('XML nuevo :', xml_out)
print('Imagen    :', img_out)
```

### Verificación: releer el XML nuevo

```python
_, new_boxes, _ = parse_voc_xml(xml_out)          # tu parser sigue funcionando
print('Barcos en XML nuevo:', len(new_boxes))

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
draw_boxes(ax, BOXES, 'lime', 1.2)                # original
draw_boxes(ax, new_boxes, 'red', 1.2)             # SAM
ax.set_title('Verde: caja original · Rojo: caja ajustada por SAM'); ax.axis('off')
plt.show()

print(open(xml_out).read()[:1500])
```

### (Opcional) Procesar todo el dataset

```python
def process_scene(scene, limit=None):
    img_dir, lbl_dir = SCENES[scene]
    xmls = sorted(glob.glob(os.path.join(DATA_ROOT, lbl_dir, '*.xml')))[:limit]
    log = []
    for xp in xmls:
        _, boxes, _ = parse_voc_xml(xp)
        if not boxes:
            continue
        ip = resolve_image_path(img_dir, xp)
        img = np.array(Image.open(ip).convert('RGB'))
        res = segment_ships_with_boxes(img, boxes, predictor)
        write_sam_xml(xp, xp.replace(DATA_ROOT, DATA_ROOT_SAM, 1), res)
        dst = ip.replace(DATA_ROOT, DATA_ROOT_SAM, 1)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(ip, dst)
        log.append((os.path.basename(xp), len(boxes)))
    return log

# process_scene('multi', limit=10)   # prueba pequeña antes de correr todo
```

---

## 6. Análisis y conclusiones (punto 8)

> Completa con **tus** números (tablas `rec` y `metr`) y tus observaciones visuales. La siguiente estructura te sirve de guía.

### 6.1 SAM automático sobre el recorte

- Número de barcos en el recorte y porcentaje recuperado (`rec.detectado.mean()`).
- Cantidad de máscaras que **no** son barcos (mar, olas, estelas): SAM no tiene noción de clase, por lo que segmenta todo lo que tenga bordes coherentes y requiere filtrado posterior.
- Casos de sobre-segmentación o fusión barco+estela.

### 6.2 SAM con bounding box como prompt

- La caja resuelve la ambigüedad principal ("¿qué objeto quiero?"): se obtiene **una máscara por barco**, sin ruido de fondo y sin necesidad de filtrar.
- Comparar `relleno` medio: indica cuánto "aire" (agua) incluían las cajas originales y cuánto mejora el ajuste.
- Reportar score medio de SAM, barcos con score bajo y casos fallidos (barcos diminutos de pocos píxeles, bajo contraste con el agua, estelas, sombras, barcos pegados donde la máscara invade al vecino).

### 6.3 Comparación entre ambos enfoques

| Aspecto | SAM automático | SAM con caja |
|---|---|---|
| Requiere anotación previa | No | Sí (cajas del XML) |
| Selecciona solo barcos | No | Sí |
| Costo | Alto (muchos prompts por imagen) | Bajo (un prompt por barco) |
| Precisión del contorno | Variable | Alta cuando la caja es correcta |
| Uso ideal | Exploración / proponer objetos | Refinar etiquetas existentes |

### 6.4 Valor del dataset `_SAM`

- Se pasa de **detección** (cajas) a **segmentación** (polígonos) sin anotar a mano; permite entrenar modelos de segmentación de instancias (Mask R-CNN, YOLO-seg).
- Cajas más ajustadas → menos ruido de fondo en el entrenamiento.
- **Riesgos:** errores de SAM se propagan como etiquetas incorrectas (*label noise*); conviene revisión manual de objetos con score bajo o `relleno` atípico, y SAM depende de la calidad de la caja inicial.

### 6.5 Posibles mejoras

- Usar `multimask_output=True` y elegir la máscara por score o por coherencia con el área esperada.
- Añadir un **punto positivo** en el centro de la caja para casos ambiguos.
- Filtrar por score / relleno y marcar los casos dudosos para revisión.
- Probar modelos adaptados a teledetección (p. ej. variantes de SAM ajustadas a imágenes satelitales) o `vit_l`/`vit_b` para comparar velocidad vs. calidad.
- Evaluar cuantitativamente con una pequeña muestra segmentada manualmente (IoU de máscara real vs. SAM).

---

## Checklist de entrega

- [ ] Explicación de SAM (sección 0)
- [ ] Imagen con ≥ 20 barcos mostrada con sus cajas
- [ ] Recorte + SAM automático + análisis
- [ ] SAM por caja sobre la imagen completa (visualización + métricas)
- [ ] Carpeta `MASATI-v2_SAM` con XML actualizados (y verificación de lectura)
- [ ] Conclusiones con tus resultados
