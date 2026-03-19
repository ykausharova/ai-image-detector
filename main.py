import sys
from detector import load_detectors, analyze_image, combined_verdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <image_path>")
        return

    image_path = sys.argv[1]
    print(f"\nAnalyzing: {image_path}\n")

    print("Loading models...")
    detectors = load_detectors()

    results = analyze_image(image_path, detectors)

    for model_name, scores in results.items():
        print(f"[ {model_name} ]")
        for label, score in scores.items():
            print(f"  {label}: {score}")
        print()

    verdict = combined_verdict(results)
    print(f"VERDICT: {verdict['verdict']} ({verdict['confidence']*100:.1f}% confidence)")

if __name__ == "__main__":
    main()