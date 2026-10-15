// =============================================================================
// TCNet Dashboard — Root Layout
// Configures dark theme, Inter font, and global page shell.
// =============================================================================

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TCNet Dashboard — Tropical Cyclone Intensity Estimator",
  description:
    "Real-time tropical cyclone intensity estimation using deep learning on IR satellite imagery. Powered by a PyTorch CNN and FastAPI backend.",
  keywords: [
    "cyclone",
    "tropical cyclone",
    "intensity estimation",
    "deep learning",
    "satellite",
    "IMD",
    "PyTorch",
    "TCNet",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased`}>
        <div className="flex min-h-screen flex-col">{children}</div>
      </body>
    </html>
  );
}
