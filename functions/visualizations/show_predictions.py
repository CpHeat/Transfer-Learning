def show_predictions(X, y_pred, y_true=None, class_names=None, n_images=10):
    import matplotlib.pyplot as plt
    import numpy as np

    n_images = min(n_images, len(X), 10)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_images):
        ax = axes[i]
        img = X[i]

        pred_class_idx = np.argmax(y_pred[i])
        pred_class_name = class_names[pred_class_idx] if class_names else str(pred_class_idx)
        pred_prob = y_pred[i][pred_class_idx]

        if y_true is not None:
            true_class_idx = np.argmax(y_true[i])
            true_class_name = class_names[true_class_idx] if class_names else str(true_class_idx)
        else:
            true_class_name = None

        ax.imshow(img.astype('uint8'))
        ax.axis('off')

        title = f"Prédit: {pred_class_name} ({pred_prob:.2f})"
        if true_class_name is not None:
            title += f"\nVrai: {true_class_name}"
        ax.set_title(title, fontsize=10)

    for j in range(n_images, 10):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()