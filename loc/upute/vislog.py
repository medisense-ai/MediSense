import re
import matplotlib.pyplot as plt

def parse_training_log(log_filename="training.log"):
    """
    Parse the training log file to extract epoch-level metrics:
    training loss, training IoU, validation loss, and validation IoU.
    Returns lists of epochs and corresponding metrics, as well as final test metrics.
    """
    epochs = []
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    test_loss = None
    test_iou = None

    # Updated regex patterns: allow any characters before "Epoch" or "Test"
    train_pattern = re.compile(r".*Epoch \[(\d+)/\d+\]\s+Training Loss: ([\d\.]+), Training IoU: ([\d\.]+)")
    val_pattern = re.compile(r".*Epoch \[(\d+)/\d+\]\s+Validation Loss: ([\d\.]+), Validation IoU: ([\d\.]+)")
    test_pattern = re.compile(r".*Test Loss: ([\d\.]+), Test IoU: ([\d\.]+)")

    with open(log_filename, "r") as f:
        for line in f:
            # Skip lines that contain 'Batch'
            if "Batch" in line:
                continue

            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                iou = float(train_match.group(3))
                epochs.append(epoch)
                train_losses.append(loss)
                train_ious.append(iou)
                continue  # once matched, continue to next line

            val_match = val_pattern.search(line)
            if val_match:
                loss = float(val_match.group(2))
                iou = float(val_match.group(3))
                val_losses.append(loss)
                val_ious.append(iou)
                continue

            test_match = test_pattern.search(line)
            if test_match:
                test_loss = float(test_match.group(1))
                test_iou = float(test_match.group(2))
    
    return epochs, train_losses, train_ious, val_losses, val_ious, test_loss, test_iou

def plot_training_metrics(log_filename="training.log"):
    """
    Read the training log and plot training loss, training IoU,
    validation loss, and validation IoU over epochs.
    Also prints the final test loss and IoU.
    """
    epochs, train_losses, train_ious, val_losses, val_ious, test_loss, test_iou = parse_training_log(log_filename)
    
    if not epochs:
        print("No epoch-level metrics found in the log.")
        return
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    axs[0].plot(epochs, train_losses, marker='o', label="Training Loss")
    axs[0].plot(epochs, val_losses, marker='o', label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss over Epochs")
    axs[0].legend()
    
    axs[1].plot(epochs, train_ious, marker='o', label="Training IoU")
    axs[1].plot(epochs, val_ious, marker='o', label="Validation IoU")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("IoU")
    axs[1].set_title("IoU over Epochs")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    if test_loss is not None and test_iou is not None:
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final Test IoU: {test_iou:.4f}")
    else:
        print("Test metrics not found in the log.")

def main():
    plot_training_metrics(log_filename="training.log")

if __name__ == "__main__":
    main()
