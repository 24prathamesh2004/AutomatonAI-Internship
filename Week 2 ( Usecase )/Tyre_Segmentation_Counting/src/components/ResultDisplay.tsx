import type { SegmentationResult } from "@/pages/Index";
import { Card } from "@/components/ui/card";

interface ResultDisplayProps {
  result: SegmentationResult;
}

export const ResultDisplay = ({ result }: ResultDisplayProps) => {
  return (
    <div className="border-2 border-foreground p-6 bg-card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">RESULTS</h2>
        <div className="bg-foreground text-background px-4 py-2 font-mono font-bold text-lg">
          COUNT: {result.count}
        </div>
      </div>

      <div className="border-2 border-foreground">
        <img
          src={`data:image/png;base64,${result.image_base64}`}
          alt="Segmented result"
          className="w-full object-contain"
        />
      </div>

      {result.detections.length > 0 && (
        <div className="mt-4">
          <h3 className="font-bold mb-2 font-mono">DETECTIONS</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {result.detections.map((det, idx) => (
              <Card
                key={idx}
                className="p-3 border-2 border-foreground bg-secondary"
              >
                <p className="font-mono text-sm">
                  <span className="text-muted-foreground">#{idx + 1}</span>
                </p>
                <p className="font-mono text-sm">
                  Conf: {(det.confidence * 100).toFixed(1)}%
                </p>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
