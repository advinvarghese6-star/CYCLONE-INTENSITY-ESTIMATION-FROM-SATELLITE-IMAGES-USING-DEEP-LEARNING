// =============================================================================
// TCNet Dashboard — Control Panel Component
// Tabbed interface embedding Upload & Preset tabs, rotation toggle, and CTA.
// =============================================================================

"use client";

import React, { useState, useCallback } from "react";
import { Zap, Info, Loader2, UploadCloud, Image } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import UploadDropzone from "@/components/upload-dropzone";
import PresetGallery from "@/components/preset-gallery";

interface ControlPanelProps {
  isLoading: boolean;
  onRunUpload: (file: File, useRotationBlending: boolean) => void;
  onRunPreset: (presetId: string, useRotationBlending: boolean) => void;
}

export default function ControlPanel({
  isLoading,
  onRunUpload,
  onRunPreset,
}: ControlPanelProps) {
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState<File | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [rotationBlending, setRotationBlending] = useState(false);

  const handleRun = useCallback(() => {
    if (activeTab === "upload" && file) {
      onRunUpload(file, rotationBlending);
    } else if (activeTab === "presets" && selectedPreset) {
      onRunPreset(selectedPreset, rotationBlending);
    }
  }, [activeTab, file, selectedPreset, rotationBlending, onRunUpload, onRunPreset]);

  const canRun =
    (activeTab === "upload" && file !== null) ||
    (activeTab === "presets" && selectedPreset !== null);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <div className="rounded-lg bg-cyan-500/10 p-1.5">
            <UploadCloud className="h-4 w-4 text-cyan-400" />
          </div>
          Input Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* ── Tabbed Input ─────────────────────────────────── */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full">
            <TabsTrigger value="upload" className="flex-1 gap-1.5">
              <UploadCloud className="h-3.5 w-3.5" />
              Upload NPY / Image
            </TabsTrigger>
            <TabsTrigger value="presets" className="flex-1 gap-1.5">
              <Image className="h-3.5 w-3.5" />
              Preset Gallery
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <UploadDropzone file={file} onFileChange={setFile} />
          </TabsContent>

          <TabsContent value="presets">
            <PresetGallery
              selectedPreset={selectedPreset}
              onSelectPreset={setSelectedPreset}
            />
          </TabsContent>
        </Tabs>

        {/* ── Rotation Blending Toggle ─────────────────────── */}
        <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-800/30 p-3.5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-200">
              Rotation Blending
            </span>
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="cursor-pointer">
                    <Info className="h-3.5 w-3.5 text-zinc-500 hover:text-zinc-300 transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="max-w-[220px]">
                    <strong>10-Angle Test-Time Augmentation (TTA)</strong>
                    <br />
                    Rotates the input image at 10 evenly-spaced angles (0°–360°),
                    runs inference on each, and averages predictions for a more
                    robust intensity estimate. Increases latency by ~10×.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Switch
            checked={rotationBlending}
            onCheckedChange={setRotationBlending}
          />
        </div>

        {/* ── Run Button ───────────────────────────────────── */}
        <Button
          onClick={handleRun}
          disabled={!canRun || isLoading}
          className="w-full"
          size="lg"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyzing Satellite Data…
            </>
          ) : (
            <>
              <Zap className="h-4 w-4" />
              Run Estimation
            </>
          )}
        </Button>

        {/* Subtle helper text */}
        {!canRun && !isLoading && (
          <p className="text-center text-xs text-zinc-600">
            {activeTab === "upload"
              ? "Upload a satellite image to enable estimation"
              : "Select a preset sample to enable estimation"}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
