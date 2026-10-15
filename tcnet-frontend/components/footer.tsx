// =============================================================================
// TCNet Dashboard — Footer Component
// Disclaimer banner & copyright.
// =============================================================================

"use client";

import React, { useState, useEffect } from "react";


export default function Footer() {
  const [year, setYear] = useState(2026);
  useEffect(() => {
    setYear(new Date().getFullYear());
  }, []);

  return (
    <footer className="border-t border-zinc-800/60 bg-zinc-950/60 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {/* Bottom Bar */}
        <div className="mt-4 flex flex-col items-center justify-between gap-2 text-xs text-zinc-500 sm:flex-row">
          <span>
            © {year} TCNet Dashboard — Built with Next.js,
            PyTorch &amp; FastAPI
          </span>
          <span className="text-zinc-600">v1.0.0</span>
        </div>
      </div>
    </footer>
  );
}
