import torch

def main():
    print("Hello from custom-train!")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    # Load your YOLO dataset


if __name__ == "__main__":
    main()
