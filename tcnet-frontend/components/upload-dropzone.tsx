// =============================================================================
// TCNet Dashboard — Upload Dropzone Component
// Drag-and-drop file upload zone for .npy and image files.
// =============================================================================

"use client";

import React, { useCallback, useRef, useState } from "react";
import { UploadCloud, FileImage, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface UploadDropzoneProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export default function UploadDropzone({ file, onFileChange }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const ACCEPTED_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "application/octet-stream", // .npy files
  ];

  const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"];

  const isValidFile = useCallback(
    (f: File) => {
      const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
      return ACCEPTED_TYPES.includes(f.type) || ACCEPTED_EXTENSIONS.includes(ext);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped && isValidFile(dropped)) {
        onFileChange(dropped);
      }
    },
    [isValidFile, onFileChange]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) onFileChange(selected);
    },
    [onFileChange]
  );

  const removeFile = useCallback(() => {
    onFileChange(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [onFileChange]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-3">
      {/* Dropzone area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-all duration-300",
          isDragging
            ? "border-cyan-400 bg-cyan-500/10 shadow-lg shadow-cyan-500/10"
            : "border-zinc-700 bg-zinc-800/30 hover:border-zinc-600 hover:bg-zinc-800/50",
          file && "border-emerald-500/40 bg-emerald-500/5"
        )}
      >
        {!file ? (
          <>
            <div
              className={cn(
                "rounded-xl p-3 transition-colors",
                isDragging ? "bg-cyan-500/20" : "bg-zinc-800"
              )}
            >
              <UploadCloud
                className={cn(
                  "h-8 w-8 transition-colors",
                  isDragging ? "text-cyan-400" : "text-zinc-400"
                )}
              />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-zinc-300">
                Drop satellite image here, or{" "}
                <span className="text-cyan-400 underline underline-offset-2">
                  browse
                </span>
              </p>
              <p className="mt-1 text-xs text-zinc-500">
                Supports NPY, PNG, JPEG, TIFF — 64×64 IR satellite arrays
              </p>
            </div>
          </>
        ) : (
          <div className="flex items-center gap-3 px-4">
            <div className="rounded-lg bg-emerald-500/15 p-2">
              <FileImage className="h-5 w-5 text-emerald-400" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-zinc-200">
                {file.name}
              </p>
              <p className="text-xs text-zinc-500">{formatFileSize(file.size)}</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={(e) => {
                e.stopPropagation();
                removeFile();
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept=".npy,.png,.jpg,.jpeg,.tiff,.tif"
        className="hidden"
        onChange={handleInputChange}
      />
    </div>
  );
}
