import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageUpload } from "@/components/ImageUpload";
import { VideoUpload } from "@/components/VideoUpload";
import { ResultDisplay } from "@/components/ResultDisplay";
import { VideoResultDisplay } from "@/components/VideoResultDisplay";

export interface SegmentationResult {
  count: number;
  image_base64: string;
  detections: Array<{
    confidence: number;
    bbox: number[];
  }>;
}

export interface VideoResult {
  total_count: number;
  video_url: string;
  frame_counts: number[];
}

const Index = () => {
  const [imageResult, setImageResult] = useState<SegmentationResult | null>(null);
  const [videoResult, setVideoResult] = useState<VideoResult | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8 border-b-4 border-foreground pb-4">
          <h1 className="text-4xl font-bold tracking-tight">
            TYRE SEGMENTATION
          </h1>
          <p className="text-muted-foreground mt-2 font-mono">
            YOLOv8 Instance Segmentation • Detection • Counting • Tracking
          </p>
        </header>

        <Tabs defaultValue="image" className="w-full">
          <TabsList className="w-full border-2 border-foreground bg-background mb-6">
            <TabsTrigger
              value="image"
              className="flex-1 data-[state=active]:bg-foreground data-[state=active]:text-background font-semibold"
            >
              IMAGE
            </TabsTrigger>
            <TabsTrigger
              value="video"
              className="flex-1 data-[state=active]:bg-foreground data-[state=active]:text-background font-semibold"
            >
              VIDEO
            </TabsTrigger>
          </TabsList>

          <TabsContent value="image" className="space-y-6">
            <ImageUpload
              onResult={setImageResult}
              loading={loading}
              setLoading={setLoading}
            />
            {imageResult && <ResultDisplay result={imageResult} />}
          </TabsContent>

          <TabsContent value="video" className="space-y-6">
            <VideoUpload
              onResult={setVideoResult}
              loading={loading}
              setLoading={setLoading}
            />
            {videoResult && <VideoResultDisplay result={videoResult} />}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
