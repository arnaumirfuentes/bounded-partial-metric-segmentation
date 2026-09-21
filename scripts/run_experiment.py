import pydicom
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage import morphology, io
from scipy.ndimage import binary_fill_holes
from pathlib import Path

# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

# Fuzzy C-means
K = 5
M = 2
MAX_ITER = 100
TOL = 1e-5

# Reproducibility
RANDOM_STATE = 42

# Product-space aggregation
GRAY_WEIGHT = 7.0

# Partial metric parameters
PARTIAL_C = 0.2
PARTIAL_T = 0.4

# Liver-cluster selection
BODY_THRESHOLD = 0.03
EROSION_FRACTION = 0.05
MIN_EROSION_RADIUS = 5

# Morphological post-processing
CLOSING_SIZE = 3

METHODS = [
    "classic_no_aggregation",
    "classic_aggregation",
    "partial_aggregation"
]

SHOW_FIGURES = True
SAVE_MASKS = True

# ============================================================
# DATASET
# ============================================================

# Repository-local data directory. The MRI data and reference masks are not
# distributed in this repository; see data/README.md.
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

GROUND_TRUTH_DIR = (
    BASE_DIR
    / "PixelLabelData"
    / "pixelLabelData"
)

CASES = [
    {
        "id": "ST003-SE005-MR022",
        "mri": BASE_DIR / "ST003-SE005-MR022",
        "ground_truth":
            GROUND_TRUTH_DIR
            / "Label_1_ST003-SE005-MR0022_MRI.png"
    },
    {
        "id": "ST003-SE007-MR018",
        "mri": BASE_DIR / "ST003-SE007-MR018",
        "ground_truth":
            GROUND_TRUTH_DIR
            / "Label_2_ST003-SE007-MR0018_MRI.png"
    },
    {
        "id": "ST004-SE007-MR021",
        "mri": BASE_DIR / "ST004-SE007-MR021",
        "ground_truth":
            GROUND_TRUTH_DIR
            / "Label_3_ST004-SE007-MR0021_MRI.png"
    }
]

def main():

    # Crear carpeta general de resultados
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_results = []

    # =========================================================
    # RECORRER TODAS LAS IMÁGENES
    # =========================================================

    for case in CASES:

        case_id = case["id"]

        print("\n" + "#" * 70)
        print("IMAGE:", case_id)
        print("#" * 70)

        # -----------------------------------------------------
        # READ MRI AND GROUND TRUTH
        # -----------------------------------------------------

        dicom_image = read_dicom_image(
            case["mri"]
        )

        ground_truth = io.imread(
            case["ground_truth"],
            as_gray=True
        )

        # Ajustar GT al tamaño de la MRI
        _, ground_truth = resize_ground_truth(
            dicom_image,
            ground_truth
        )

        # Carpeta específica de esta imagen
        case_results_dir = (
            RESULTS_DIR / case_id
        )

        case_results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # COMPROBACIÓN VISUAL MRI + GT
        # -----------------------------------------------------

        if SHOW_FIGURES:

            plt.figure(
                figsize=(6, 6)
            )

            plt.imshow(
                dicom_image,
                cmap="gray"
            )

            plt.imshow(
                ground_truth,
                cmap="Reds",
                alpha=0.35
            )

            plt.axis("off")

            plt.title(
                f"{case_id}\nMRI + Ground Truth"
            )

            plt.tight_layout()

            plt.show()

        # Guardar overlay para reproducibilidad
        plt.figure(
            figsize=(6, 6)
        )

        plt.imshow(
            dicom_image,
            cmap="gray"
        )

        plt.imshow(
            ground_truth,
            cmap="Reds",
            alpha=0.35
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            case_results_dir / "ground_truth_overlay.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Resultados de los tres métodos para esta imagen
        case_method_results = {}

        # =====================================================
        # RECORRER LOS TRES MÉTODOS
        # =====================================================

        for method in METHODS:

            print("\n" + "=" * 60)
            print("METHOD:", method)
            print("=" * 60)

            # ---------------------------------------------
            # FUZZY C-MEANS
            # ---------------------------------------------

            clustered_image = k_fuzzy_means(
                dicom_image,
                method=method
            )

            # ---------------------------------------------
            # LIVER CLUSTER SELECTION
            # ---------------------------------------------

            liver_cluster_mask = (
                select_liver_cluster(
                    clustered_image,
                    dicom_image
                )
            )

            # ---------------------------------------------
            # LARGEST CONNECTED COMPONENT
            # ---------------------------------------------

            largest_component_mask = (
                find_largest_connected_component(
                    liver_cluster_mask
                )
            )

            # ---------------------------------------------
            # POST-PROCESSING
            # ---------------------------------------------

            final_mask = fill_holes(
                largest_component_mask
            )

            # ---------------------------------------------
            # EVALUATION
            # ---------------------------------------------

            metrics = evaluate_segmentation(
                final_mask,
                ground_truth
            )

            # Guardar resultados en memoria
            case_method_results[method] = {
                "clustered": clustered_image,
                "mask": final_mask,
                "metrics": metrics
            }

            # Añadir fila a la tabla global
            evaluation_results.append({
                "Image": case_id,
                "Method": method,
                **metrics
            })

            # ---------------------------------------------
            # TERMINAL OUTPUT
            # ---------------------------------------------

            print("\nEvaluation:")

            print(
                "TP:",
                metrics["TP"]
            )

            print(
                "TN:",
                metrics["TN"]
            )

            print(
                "FP:",
                metrics["FP"]
            )

            print(
                "FN:",
                metrics["FN"]
            )

            print(
                "Misclassified:",
                metrics["Misclassified"]
            )

            print(
                "Misclassification Rate:",
                f'{metrics["Misclassification Rate (%)"]:.2f}%'
            )

            print(
                "Accuracy:",
                f'{metrics["Accuracy (%)"]:.2f}%'
            )

            print(
                "Dice:",
                f'{metrics["Dice"]:.4f}'
            )

            print(
                "IoU:",
                f'{metrics["IoU"]:.4f}'
            )

            # ---------------------------------------------
            # SAVE MASK
            # ---------------------------------------------

            if SAVE_MASKS:

                mask_path = (
                    case_results_dir
                    / f"{method}_mask.png"
                )

                io.imsave(
                    mask_path,
                    (
                        final_mask.astype(
                            np.uint8
                        ) * 255
                    ),
                    check_contrast=False
                )

        # =====================================================
        # FIGURA COMPARATIVA DE ESTA MRI
        # =====================================================

        fig, axes = plt.subplots(
            len(METHODS),
            4,
            figsize=(15, 12)
        )

        for i, method in enumerate(METHODS):

            # MRI
            axes[i, 0].imshow(
                dicom_image,
                cmap="gray"
            )

            axes[i, 0].set_title(
                "Original MRI"
            )

            # Clustering
            axes[i, 1].imshow(
                case_method_results[
                    method
                ]["clustered"],
                cmap="gray"
            )

            axes[i, 1].set_title(
                f"{method}\nClustering"
            )

            # Final mask
            axes[i, 2].imshow(
                case_method_results[
                    method
                ]["mask"],
                cmap="gray"
            )

            axes[i, 2].set_title(
                f"{method}\nFinal mask"
            )

            # Ground truth
            axes[i, 3].imshow(
                ground_truth,
                cmap="gray"
            )

            dice = (
                case_method_results[
                    method
                ]["metrics"]["Dice"]
            )

            iou = (
                case_method_results[
                    method
                ]["metrics"]["IoU"]
            )

            axes[i, 3].set_title(
                f"Ground truth\n"
                f"Dice={dice:.3f}, "
                f"IoU={iou:.3f}"
            )

            for j in range(4):
                axes[i, j].axis("off")

        plt.tight_layout()

        # Guardar comparación
        plt.savefig(
            case_results_dir
            / "comparison.png",
            dpi=300,
            bbox_inches="tight"
        )

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close()

    # =========================================================
    # GUARDAR RESULTADOS DE LOS 9 EXPERIMENTOS
    # =========================================================

    save_results_csv(
        evaluation_results,
        RESULTS_DIR
        / "segmentation_results.csv"
    )

    # =========================================================
    # TABLA RESUMEN
    # =========================================================

    save_summary_csv(
        evaluation_results,
        RESULTS_DIR
        / "summary_results.csv"
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)

    print(
        "\nDetailed results:",
        RESULTS_DIR
        / "segmentation_results.csv"
    )

    print(
        "Summary results:",
        RESULTS_DIR
        / "summary_results.csv"
    )
    
    
def read_dicom_image(filepath):
    # Read DICOM image
    dicom_data = pydicom.dcmread(filepath)
    # Extract pixel data
    image = dicom_data.pixel_array
    return image

def find_largest_connected_component(clustered_image):
    # Etiquetar las componentes conectadas en la imagen segmentada
    labeled_image = label(clustered_image)
    
    # Calcular las propiedades de las regiones etiquetadas
    regions = regionprops(labeled_image)
    # Encontrar la región con el área más grande
    max_area = 0
    max_region_label = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region_label = region.label
    
    # Crear una máscara para la región con el área más grande
    largest_component_mask = labeled_image == max_region_label
    
    return largest_component_mask

def select_liver_cluster(clustered_image, original_image):

    # Normalizar imagen
    image_norm = original_image / np.max(original_image)

    # 1. Crear una máscara aproximada del cuerpo
    body_mask = image_norm > BODY_THRESHOLD

    # Rellenar huecos internos
    body_mask = binary_fill_holes(body_mask)

    # Quedarnos con la componente corporal principal
    labeled_body = label(body_mask)
    regions = regionprops(labeled_body)

    largest_body = max(regions, key=lambda r: r.area)
    body_mask = labeled_body == largest_body.label

    # 2. Erosionar el cuerpo para eliminar tejidos periféricos
    #    como grasa subcutánea y pared abdominal
    radius = max(
        MIN_EROSION_RADIUS,
        int(
            EROSION_FRACTION
            * min(clustered_image.shape)
            )
    )
    inner_body = morphology.erosion(
    body_mask,
    morphology.disk(radius)
)

    # 3. Evaluar cada cluster
    unique_labels = np.unique(clustered_image)

    best_label = None
    best_score = -1

    for lab in unique_labels:

        cluster_mask = clustered_image == lab

        # Componentes conexas del cluster
        labeled_cluster = label(cluster_mask)
        regions = regionprops(labeled_cluster)

        if len(regions) == 0:
            continue

        # Componente conexa más grande
        largest_region = max(regions, key=lambda r: r.area)
        component = labeled_cluster == largest_region.label

        # Cuánto de esa componente está realmente dentro del cuerpo
        overlap = np.sum(component & inner_body)

        # Fracción de la componente situada en la zona interna
        fraction_inside = overlap / np.sum(component)

        # Score: tamaño interno + penalización si está en la periferia
        score = overlap * fraction_inside

        print(
            "Cluster:", lab,
            "area:", np.sum(component),
            "internal overlap:", overlap,
            "fraction inside:", fraction_inside,
            "score:", score
        )

        if score > best_score:
            best_score = score
            best_label = lab

    print("Selected liver cluster:", best_label)

    return clustered_image == best_label





def k_fuzzy_means(
    image,
    method="partial_aggregation",
    k=K,
    m=M,
    max_iter=MAX_ITER,
    tol=TOL
):

    # Normalizar intensidad
    image_norm = image / np.max(image)

    rows, cols = image_norm.shape

    # ---------------------------------------------------------
    # REPRESENTACIÓN DE LOS DATOS
    # ---------------------------------------------------------

    if method == "classic_no_aggregation":

        # Solo nivel de gris
        data = image_norm.ravel().reshape(-1, 1)

    else:

        # Posición normalizada
        positions = np.array([
            [i / rows, j / cols]
            for i in range(rows)
            for j in range(cols)
        ])

        # Posición + intensidad
        data = np.column_stack([
            positions,
            image_norm.ravel()
        ])

    # ---------------------------------------------------------
    # INICIALIZACIÓN
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",
        random_state=RANDOM_STATE
    )

    kmeans.fit(data)
    centers = kmeans.cluster_centers_

    # ---------------------------------------------------------
    # FUZZY C-MEANS
    # ---------------------------------------------------------

    for _ in range(max_iter):

        if method == "classic_no_aggregation":

            distances_to_centers = (
                distances_classic_no_aggregation(
                    data,
                    centers
                )
            )

        elif method == "classic_aggregation":

            distances_to_centers = (
                distances_classic_aggregation(
                    data,
                    centers
                )
            )

        elif method == "partial_aggregation":

            distances_to_centers = (
                distances_partial_aggregation(
                    data,
                    centers
                )
            )

        else:

            raise ValueError(
                "Unknown method: " + method
            )

        # Evitar divisiones por cero
        distances_to_centers = np.maximum(
            distances_to_centers,
            1e-12
        )

        # Memberships
        membership = (
            1 /
            distances_to_centers**(2 / (m - 1))
        )

        membership_norm = (
            membership /
            np.sum(
                membership,
                axis=1,
                keepdims=True
            )
        )

        # Nuevos centroides
        weights = membership_norm**m

        new_centers = (
            np.sum(
                weights[:, :, np.newaxis]
                * data[:, np.newaxis, :],
                axis=0
            )
            /
            np.sum(
                weights,
                axis=0
            )[:, np.newaxis]
        )

        # Convergencia
        if np.linalg.norm(
            centers - new_centers
        ) < tol:

            centers = new_centers
            break

        centers = new_centers

    # Recalcular memberships con los centros finales
    if method == "classic_no_aggregation":

        distances_to_centers = (
            distances_classic_no_aggregation(
                data,
                centers
            )
        )

    elif method == "classic_aggregation":

        distances_to_centers = (
            distances_classic_aggregation(
                data,
                centers
            )
        )

    else:

        distances_to_centers = (
            distances_partial_aggregation(
                data,
                centers
            )
        )

    distances_to_centers = np.maximum(
        distances_to_centers,
        1e-12
    )

    membership = (
        1 /
        distances_to_centers**(2 / (m - 1))
    )

    membership_norm = (
        membership /
        np.sum(
            membership,
            axis=1,
            keepdims=True
        )
    )

    labels = np.argmax(
        membership_norm,
        axis=1
    )

    return labels.reshape(image.shape)

def fill_holes(binary_mask):

    # Rellenar agujeros internos
    filled_mask = binary_fill_holes(binary_mask)

    # Suavizar pequeños huecos e irregularidades del borde
    final_mask = morphology.closing(
        filled_mask,
        morphology.square(CLOSING_SIZE)
    )

    return final_mask

def resize_ground_truth(segmented_image, ground_truth_image):

    if segmented_image.shape != ground_truth_image.shape:

        print(
            "Resizing ground truth from",
            ground_truth_image.shape,
            "to",
            segmented_image.shape
        )

        ground_truth_image = cv2.resize(
            ground_truth_image,
            (
                segmented_image.shape[1],
                segmented_image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    return segmented_image, ground_truth_image


def evaluate_segmentation(segmented_image, ground_truth_image):

    # Asegurar mismas dimensiones
    segmented_image, ground_truth_image = resize_ground_truth(
    segmented_image,
    ground_truth_image
)
    # Convertir ambas imágenes a máscaras binarias
    segmented_image = segmented_image > 0
    ground_truth_image = ground_truth_image > 0

    # True Positive
    TP = np.sum(
        segmented_image & ground_truth_image
    )

    # True Negative
    TN = np.sum(
        (~segmented_image) & (~ground_truth_image)
    )

    # False Positive
    FP = np.sum(
        segmented_image & (~ground_truth_image)
    )

    # False Negative
    FN = np.sum(
        (~segmented_image) & ground_truth_image
    )

    total_pixels = ground_truth_image.size

    total_misclassified = FP + FN

    misclassification_rate = (
        total_misclassified / total_pixels
    ) * 100

    accuracy = (
        (TP + TN) / total_pixels
    ) * 100

    # Dice coefficient
    dice_denominator = 2 * TP + FP + FN

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * TP / dice_denominator
        )

    # Intersection over Union / Jaccard
    iou_denominator = TP + FP + FN

    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = (
            TP / iou_denominator
        )

    return {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "Misclassified": int(total_misclassified),
        "Misclassification Rate (%)": float(
            misclassification_rate
        ),
        "Accuracy (%)": float(accuracy),
        "Dice": float(dice),
        "IoU": float(iou)
    }

def save_results_csv(
    evaluation_results,
    filename
):

    fieldnames = [
        "Image",
        "Method",
        "TP",
        "TN",
        "FP",
        "FN",
        "Misclassified",
        "Misclassification Rate (%)",
        "Accuracy (%)",
        "Dice",
        "IoU"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in evaluation_results:
            writer.writerow(row)

def save_summary_csv(
    evaluation_results,
    filename
):

    fieldnames = [
        "Method",
        "Number of Images",
        "Mean Misclassification Rate (%)",
        "Mean Accuracy (%)",
        "Mean Dice",
        "Mean IoU"
    ]

    summary_rows = []

    for method in METHODS:

        method_results = [
            row
            for row in evaluation_results
            if row["Method"] == method
        ]

        mean_error = np.mean([
            row[
                "Misclassification Rate (%)"
            ]
            for row in method_results
        ])

        mean_accuracy = np.mean([
            row["Accuracy (%)"]
            for row in method_results
        ])

        mean_dice = np.mean([
            row["Dice"]
            for row in method_results
        ])

        mean_iou = np.mean([
            row["IoU"]
            for row in method_results
        ])

        summary_rows.append({
            "Method": method,
            "Number of Images":
                len(method_results),
            "Mean Misclassification Rate (%)":
                mean_error,
            "Mean Accuracy (%)":
                mean_accuracy,
            "Mean Dice":
                mean_dice,
            "Mean IoU":
                mean_iou
        })

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


def partial_distance(
    x,
    y,    c=PARTIAL_C,
    t=PARTIAL_T
):
    if len(x) != len(y):
        raise ValueError("The vectors x and y must have the same length")

    sum_abs_diff = sum(abs(xi - yi) for xi, yi in zip(x, y))
    sum_max = sum(max(xi, yi) for xi, yi in zip(x, y))
    
    distance = sum_abs_diff / (t + sum_max) + c
    return distance

def distances_partial_aggregation(data, centers):

    pos_distances = np.sqrt(
        np.sum(
            (
                data[:, np.newaxis, :2]
                - centers[np.newaxis, :, :2]
            )**2,
            axis=2
        )
    )

    grey_distances = np.array([
        [
            partial_distance(
                [point[2]],
                [center[2]],
            )
            for center in centers
        ]
        for point in data
    ])

    return (
    pos_distances
    + GRAY_WEIGHT * grey_distances
)
def distances_classic_aggregation(data, centers):

    pos_distances = np.sqrt(
        np.sum(
            (
                data[:, np.newaxis, :2]
                - centers[np.newaxis, :, :2]
            )**2,
            axis=2
        )
    )

    grey_distances = np.abs(
        data[:, np.newaxis, 2]
        -
        centers[np.newaxis, :, 2]
    )

    return (
    pos_distances
    + GRAY_WEIGHT * grey_distances
)
def distances_classic_no_aggregation(data, centers):

    return np.abs(
        data[:, np.newaxis, 0]
        -
        centers[np.newaxis, :, 0]
    )
if __name__ == "__main__":
    main()