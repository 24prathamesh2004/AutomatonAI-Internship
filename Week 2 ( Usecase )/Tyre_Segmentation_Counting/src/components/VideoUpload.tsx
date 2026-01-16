import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Video } from "lucide-react";
import type { VideoResult } from "@/pages/Index";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface VideoUploadProps {
  onResult: (result: VideoResult) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export const VideoUpload = ({ onResult, loading, setLoading }: VideoUploadProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      setFileName(file.name);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!fileInputRef.current?.files?.[0]) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", fileInputRef.current.files[0]);

    try {
      const response = await fetch(`${API_URL}/segment/video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Video processing failed");
      }

      const result = await response.json();
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-2 border-foreground p-6 bg-card">
      <div className="flex items-center gap-2 mb-4">
        <Video className="w-5 h-5" />
        <h2 className="text-xl font-bold">VIDEO TRACKING</h2>
      </div>

      <div
        className="border-2 border-dashed border-muted-foreground p-8 text-center cursor-pointer hover:border-foreground transition-colors"
        onClick={() => fileInputRef.current?.click()}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-2">
            <video
              src={preview}
              className="max-h-64 mx-auto object-contain"
              controls
            />
            <p className="font-mono text-sm text-muted-foreground">{fileName}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-muted-foreground">
            <Upload className="w-12 h-12" />
            <p className="font-mono">Click to upload video</p>
            <p className="text-sm">MP4, AVI, MOV supported</p>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {error && (
        <p className="text-destructive mt-4 font-mono text-sm">{error}</p>
      )}

      <Button
        onClick={handleUpload}
        disabled={!preview || loading}
        className="w-full mt-4 font-bold"
      >
        {loading ? "PROCESSING VIDEO..." : "TRACK & COUNT"}
      </Button>
    </div>
  );
};
