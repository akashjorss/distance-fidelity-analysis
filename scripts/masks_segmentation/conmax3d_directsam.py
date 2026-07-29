import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import random
import networkx as nx
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
import time
from functools import wraps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import torch.multiprocessing as mp
import cv2
import gc


# Import DirectSAM libraries
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

# Timing Decorator and Store Function
time_taken = {}
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken[func.__name__] = end_time - start_time
        return result
    return wrapper

# Function to store time taken by each component
def store_time_taken(time_taken, output_dir, scene):
    output_dir = os.path.join(output_dir, 'time')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'execution_time_{scene}.json')
    with open(output_file, 'w') as f:
        json.dump(time_taken, f, indent=4)

# ========== Mask Generation ==========
def inference_single_image(image, image_processor, model, resolution=None, pyramid_layers=0, overlap=90):
    """
    Perform inference on a single image, optionally using a pyramid of layers.
    """
    if resolution:
        image_processor.size['height'] = resolution
        image_processor.size['width'] = resolution

    def run(image, bzp=0):
        encoding = image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(model.device).to(model.dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.float().cpu()

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()

        if bzp > 0:
            probs = boundary_zero_padding(probs, p=bzp)
        return probs

    global_probs = run(image)

    if pyramid_layers > 0:
        for layer in range(1, pyramid_layers + 1):
            boxes = generate_crop_boxes(image.size, n_layers=layer, overlap=overlap)
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = image.crop(box)
                probs = run(crop, bzp=overlap)
                global_probs[y1:y2, x1:x2] += probs
        global_probs /= (pyramid_layers + 1)

    return global_probs

@timeit
def generate_direct_sam_masks(images, image_processor, model, threshold=0.25):
    """Generate masks for each image using DirectSAM."""
    cropped_images = []
    cropped_images_to_images = {}
    for image_index, image in tqdm(enumerate(images), desc="Generating masks"):
        probs = inference_single_image(
            Image.fromarray(image),
            image_processor,
            model,
            resolution=None,
            pyramid_layers=0,
        )  # returns probs of shape (H, W)

        # Apply threshold to get binary mask
        binarized = (probs < threshold).astype(np.uint8)
        num_objects, labels = cv2.connectedComponents(binarized)
        masks = [labels == i for i in range(1, labels.max() + 1)]
        masks.sort(key=lambda x: x.sum(), reverse=True)

        for idx, mask in enumerate(masks):
            # Create an RGB image with black background
            cropped_image = mask[:, :, None] * np.array(image)
            cropped_images.append(Image.fromarray(cropped_image))
            cropped_images_to_images[len(cropped_images) - 1] = image_index

        # Free up memory
        del probs, binarized, labels, masks
        torch.cuda.empty_cache()
        gc.collect()

    # Print the total number of masks produced
    total_masks = len(cropped_images)
    print(f"Total number of masks produced: {total_masks}")

    return cropped_images, cropped_images_to_images

# def delete_small_masks(masks, H, W, min_num_pixels=None):
#     """Filter out small masks based on pixel count."""
#     min_pixels = np.sqrt(H * W) if min_num_pixels is None else min_num_pixels
#     filtered_masks = []
#     for i, mask_set in enumerate(masks):
#         filtered_masks.append([mask for mask in mask_set if np.sum(mask['segmentation']) > min_pixels])
#     return filtered_masks

# ========== Data Loading ==========
def load_data(data_path, factor=8):
    """Load LLFF data and extract image properties."""
    from load_llff import load_llff_data
    imgs, poses, bds, render_poses, i_test = load_llff_data(data_path, factor=factor)
    hwf = poses[0, :3, -1]
    H, W, focal = hwf[0], hwf[1], hwf[2]
    poses = poses[:, :3, :4]
    images = (imgs * 255).astype(np.uint8)
    return images, poses, H, W

# ========== Image Processing ==========
# @timeit
# def crop_images_worker(images, masks, result_queue, start_idx, end_idx):
#     """Worker function to crop images in parallel."""
#     local_cropped_images = []
#     local_cropped_images_to_images = {}

#     for i in range(start_idx, end_idx):
#         for j, mask in enumerate(masks[i]):
#             mask_3d = np.dstack([mask['segmentation']] * 3)
#             cropped_img = images[i] * mask_3d
#             cropped_img_pil = Image.fromarray(cropped_img)
#             local_cropped_images.append(cropped_img_pil)
#             local_cropped_images_to_images[len(local_cropped_images) - 1] = i
    
#     # Put results in a queue to avoid shared state issues
#     result_queue.put((local_cropped_images, local_cropped_images_to_images))

# def crop_images_with_masks(images, masks, num_workers=4):
#     """Parallel crop images using their masks."""
#     # Split the data into chunks based on the number of workers
#     chunk_size = len(images) // num_workers
#     chunks = [(i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else len(images)) for i in range(num_workers)]
    
#     # Create a multiprocessing queue to collect results
#     result_queue = mp.Queue()

#     # Start the parallel processing
#     processes = []
#     for start_idx, end_idx in chunks:
#         p = mp.Process(target=crop_images_worker, args=(images, masks, result_queue, start_idx, end_idx))
#         processes.append(p)
#         p.start()

#     # Collect results from all workers
#     cropped_images = []
#     cropped_images_to_images = {}

#     for _ in processes:
#         local_cropped_images, local_cropped_images_to_images = result_queue.get()
#         cropped_images.extend(local_cropped_images)
#         # Update dictionary ensuring unique indices
#         offset = len(cropped_images_to_images)
#         for k, v in local_cropped_images_to_images.items():
#             cropped_images_to_images[offset + k] = v

#     # Ensure all processes have finished
#     for p in processes:
#         p.join()

    # return cropped_images, cropped_images_to_images

def generate_image_embeddings(cropped_images, img2vec, batch_size=64):
    """Generate embeddings for the images in batches."""
    img_vectors = []
    #convert to torch tensor and move to device
    for i in tqdm(range(0, len(cropped_images), batch_size), desc="Generating embeddings"):
        batch = cropped_images[i:i + batch_size]
        batch_vectors = img2vec.get_vec(batch)
        img_vectors.extend(batch_vectors)
    return np.array(img_vectors)

def calculate_cosine_distance_matrix(img_vectors):
    """Calculate the cosine distance matrix for image embeddings using float16 precision."""
    # Convert to PyTorch tensor with float16 precision
    img_vectors = torch.tensor(img_vectors, dtype=torch.float16, device=device)
    
    # Normalize the vectors
    img_vectors = img_vectors / img_vectors.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity
    cosine_similarity = torch.mm(img_vectors, img_vectors.t())
    
    # Convert cosine similarity to cosine distance
    cosine_distance = 1 - cosine_similarity
    
    # Convert back to numpy array with float64 precision
    return cosine_distance.cpu().numpy().astype(np.float64)

@timeit
def perform_clustering(distance_matrix, num_images):
    """Perform HDBSCAN clustering on the distance matrix."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric='precomputed', 
        cluster_selection_method='eom',
    )
    return clusterer.fit_predict(distance_matrix)

# ========== Pixel Contribution Calculation ==========
@timeit
def cropped_image_to_pixel_ids(cropped_image):
    """Convert cropped image mask into pixel IDs."""
    cropped_image = np.array(cropped_image)[:, :, 0]
    i, j = np.where(cropped_image > 0)
    pixel_ids = i * cropped_image.shape[1] + j
    return pixel_ids.tolist()

def calculate_pixel_contribution(current_selection, candidate_image, G):
    # Calculate the contribution of candidate_image when added to current_selection
    new_pixels = set()
    for concept in G.neighbors(candidate_image):
        if G.nodes[concept]['type'] == 'concept':
            current_pixels = set()
            # Union pixels from already selected images for this concept
            for img in current_selection:
                for bag_node in G.neighbors(img):
                    if G.nodes[bag_node]['type'] == 'pixel_bag' and bag_node in G.neighbors(concept):
                        current_pixels.update(G.nodes[bag_node]['value'])

            # Pixels from the candidate image for this concept
            candidate_pixels = set()
            for bag_node in G.neighbors(candidate_image):
                if G.nodes[bag_node]['type'] == 'pixel_bag' and bag_node in G.neighbors(concept):
                    candidate_pixels.update(G.nodes[bag_node]['value'])
            
            # New pixels contributed by candidate image
            new_pixels.update(candidate_pixels - current_pixels)
    return len(new_pixels)

@timeit
def greedy_select_images(G, k):
    all_images = [node for node in G.nodes if G.nodes[node]['type'] == 'image']
    selected_images = []
    remaining_images = set(all_images)

    while len(selected_images) < k and remaining_images:
        max_contribution = 0
        best_image = None
        
        # Evaluate each remaining image's contribution
        for image in tqdm(remaining_images, desc="Evaluating images"):
            contribution = calculate_pixel_contribution(selected_images, image, G)
            if contribution > max_contribution:
                max_contribution = contribution
                best_image = image

        if best_image != None:
            selected_images.append(best_image)
            remaining_images.remove(best_image)
        else:
            # Randomly select an image if no image contributes new pixels
            rand_index = random.choice(list(remaining_images))
            selected_images.append(rand_index)
            remaining_images.remove(rand_index)

    return selected_images

# ========== Graph Construction ==========
@timeit
def construct_graph(num_images, cropped_images, cluster_labels, cropped_images_to_images):
    """Construct a graph connecting images, masks, concepts, and pixel bags."""
    G = nx.Graph()
    for i in range(num_images):
        G.add_node(i, type="image")

    # Add mask nodes
    for i in range(len(cropped_images)):
        G.add_node(f"{cropped_images_to_images[i]}.{i}", type="mask")

    # Add concept nodes
    for i in range(len(cluster_labels)):
        if cluster_labels[i] != -1:
            G.add_node(f"concept_{cluster_labels[i]}", type="concept")
    
    # Connect the cropped images to their original images in the graph
    for i in range(len(cropped_images)):
        G.add_edge(f"{cropped_images_to_images[i]}.{i}", cropped_images_to_images[i], type="has_mask")

    # Connect cropped_images that are in the same cluster
    for i in range(len(cropped_images)):
        for j in range(i+1, len(cropped_images)):
            if cluster_labels[i] == cluster_labels[j] and cropped_images_to_images[i] != cropped_images_to_images[j]:
                G.add_edge(f"{cropped_images_to_images[i]}.{i}", f"{cropped_images_to_images[j]}.{j}", type="same_concept")

    # Connect cropped_images to the cluster_ids they belong to
    for i in range(len(cropped_images)):
        if cluster_labels[i] != -1:
            G.add_edge(f"concept_{cluster_labels[i]}", f"{cropped_images_to_images[i]}.{i}", type="has_concept")

    # Connect images to cluster_ids their cropped images belong to
    for i in range(len(cropped_images)):
        if cluster_labels[i] != -1:
            G.add_edge(cropped_images_to_images[i], f"concept_{cluster_labels[i]}", type="has_concept")

    # Connect mask pixels to the cropped images they belong to
    for i in tqdm(range(len(cropped_images)), desc="Adding pixel bags"):
        if cluster_labels[i] == -1:
            continue
        pixel_ids = cropped_image_to_pixel_ids(cropped_images[i])
        # Add pixel_bag node
        G.add_node(f"pixel_bag_{i}", type="pixel_bag", value=pixel_ids)
        G.add_edge(f"{cropped_images_to_images[i]}.{i}", f"pixel_bag_{i}", type="has_pixel_bag")
        G.add_edge(cropped_images_to_images[i], f"pixel_bag_{i}", type="has_pixel_bag")
        G.add_edge(f"concept_{cluster_labels[i]}", f"pixel_bag_{i}", type="has_pixel_bag")

    return G

# ========== Visualization ==========
def display_cluster_images(images, cluster_labels, n_clusters=5, n_images_per_cluster=5, output_file='clusters.png'):
    """Display images from randomly selected clusters and save the plot to a file."""
    fig, axs = plt.subplots(n_clusters, n_images_per_cluster, figsize=(40, 40))
    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    for i in range(n_clusters):
        if not unique_clusters:
            break
        cluster = random.choice(unique_clusters)
        unique_clusters.remove(cluster)
        cluster_indices = [index for index, label in enumerate(cluster_labels) if label == cluster]
        for j in range(n_images_per_cluster):
            if not cluster_indices:
                break
            img = images[random.choice(cluster_indices)]
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    plt.savefig(output_file)
    plt.close(fig)

def plot_camera_positions(poses, selected_images, output_file='camera_positions.png'):
    """Plot camera positions in 3D."""
    camera_positions = poses[:, :3, 3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    selected_positions = np.array([camera_positions[i] for i in selected_images])
    non_selected_positions = np.array([camera_positions[i] for i in range(len(camera_positions)) if i not in selected_images])
    
    ax.scatter(non_selected_positions[:, 0], non_selected_positions[:, 1], non_selected_positions[:, 2], color='blue', label='Non-selected')
    ax.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2], color='red', label='Selected')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(output_file)
    plt.close(fig)

@timeit
def select_frames(base_dir, scene, num_frames):
    # Load data and models
    print("Processing scene:", scene)
    data_path = os.path.join(base_dir, scene)
    images, poses, H, W = load_data(data_path, factor=4)
    
    # DirectSAM Model configuration
    checkpoint = "chendelong/DirectSAM-1800px-0424"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
    model.half()

    # Generate masks using DirectSAM and crop images
    cropped_images, cropped_images_to_images = generate_direct_sam_masks(images, image_processor, model, threshold=0.25)

    # Generate embeddings
    from img2vec_pytorch import Img2Vec
    img2vec = Img2Vec(cuda=True, model='efficientnet_b1')
    img_vectors = generate_image_embeddings(cropped_images, img2vec)

    # Perform clustering
    print("calculating distance matrix")
    distance_matrix = calculate_cosine_distance_matrix(img_vectors)
    print("performing clustering")
    cluster_labels = perform_clustering(distance_matrix, len(images))

    print("Number of clusters:", len(set(cluster_labels)))
    print("Cluster labels:", cluster_labels)
    print("Number of outliers:", Counter(cluster_labels)[-1])

    # Save cropped images into cluster directories
    output_dir = "Clusters/directsam"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(cropped_images)):
        if cluster_labels[i] == -1:
            # Reassign outliers to the nearest cluster
            nearest_cluster = np.argmin(distance_matrix[i][cluster_labels != -1])
            cluster_labels[i] = cluster_labels[nearest_cluster]

        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_labels[i]}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        cropped_image_path = os.path.join(cluster_dir, f"cropped_image_{i}.png")
        cropped_image = Image.fromarray(np.array(cropped_images[i]))
        cropped_image.save(cropped_image_path)

    # Display images in clusters
    display_cluster_images(cropped_images, cluster_labels, output_file='clusters.png')

    G = construct_graph(len(images), cropped_images, cluster_labels, cropped_images_to_images)

    # Greedily select k images based on unique pixel contributions
    selected_images = greedy_select_images(G, num_frames)
    print("Selected images:", selected_images)
    # Display the selected images
    fig, axs = plt.subplots(4, 5, figsize=(40, 40))
    for i in range(min(20, len(selected_images))):
        img = images[selected_images[i]]
        axs[i // 5, i % 5].imshow(img)
        axs[i // 5, i % 5].axis('off')
    plt.savefig('selected_images.png')
    plt.close(fig)

    # Plot camera positions of the selected images
    # plot_camera_positions(poses, selected_images)

    return selected_images

# Helper function to convert numbers to words
def number_to_words(n):
    words = {
        10: 'ten',
        20: 'twenty',
        25: 'twenty_five'
    }
    return words.get(n, str(n))

# Store train indices in JSON file
def store_train_indices(train_indices, output_dir, scene):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Convert numpy types to native Python types
    output_file = os.path.join(output_dir, f"train_indices.json")
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            train_indices_file = json.load(f)
    else:
        train_indices_file = {}
    
    # Update the train_indices for the given scene
    train_indices_file[scene] = train_indices

    with open(output_file, 'w') as f:
        json.dump(train_indices_file, f, indent=4)

# ===

def main():
    parser = argparse.ArgumentParser(description='Generate train indices for different methods and frame counts.')
    parser.add_argument('base_dir', type=str, help='Base directory containing the scene directories')
    parser.add_argument('scene', type=str, help='Scene directory to process')
    parser.add_argument('output_dir', type=str, help='Output directory to store the JSON files')

    args = parser.parse_args()
    base_dir = args.base_dir
    output_base_dir = args.output_dir
    scene = args.scene
    frame_counts = [10, 20, 25]

    num_frames = 25  # Maximum number of frames to select
    train_indices = select_frames(base_dir, scene, num_frames)
    
    for count in frame_counts:
        train_indices_count = train_indices[:count]
        count_str = number_to_words(count)
        output_dir = os.path.join(output_base_dir, count_str, 'conmax3d/directsam')
        store_train_indices(train_indices_count, output_dir, scene)
    
    store_time_taken(time_taken, output_base_dir, scene)

# ========== Main Execution ==========
if __name__ == "__main__":
    main()

"""
Example usage:
python conmax3d_directsam.py /gpfs/workdir/malhotraa/gsplat/examples/data/360_v2 bicycle /gpfs/workdir/malhotraa/ConMax3D_experiments/gsplat/360_v2/
"""