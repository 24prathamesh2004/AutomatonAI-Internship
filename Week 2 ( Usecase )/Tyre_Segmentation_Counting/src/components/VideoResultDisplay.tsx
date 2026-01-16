import type { VideoResult } from "@/pages/Index";

interface VideoResultDisplayProps {
  result: VideoResult;
}

export const VideoResultDisplay = ({ result }: VideoResultDisplayProps) => {
  return (
    <div className="border-2 border-foreground p-6 bg-card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">VIDEO RESULTS</h2>
        <div className="bg-foreground text-background px-4 py-2 font-mono font-bold text-lg">
          TOTAL UNIQUE: {result.total_count}
        </div>
      </div>

      <div className="border-2 border-foreground">
        <video
          src={result.video_url}
          controls
          className="w-full"
        />
      </div>

      {result.frame_counts.length > 0 && (
        <div className="mt-4">
          <h3 className="font-bold mb-2 font-mono">FRAME-BY-FRAME</h3>
          <div className="flex gap-1 flex-wrap">
            {result.frame_counts.slice(0, 50).map((count, idx) => (
              <div
                key={idx}
                className="w-8 h-8 border border-foreground flex items-center justify-center font-mono text-xs bg-secondary"
                title={`Frame ${idx + 1}`}
              >
                {count}
              </div>
            ))}
            {result.frame_counts.length > 50 && (
              <div className="w-8 h-8 flex items-center justify-center font-mono text-xs text-muted-foreground">
                +{result.frame_counts.length - 50}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
