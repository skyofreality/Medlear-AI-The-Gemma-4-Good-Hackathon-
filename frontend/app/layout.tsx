import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedLearn AI",
  description: "Offline medical education assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <script
          type="importmap"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              imports: {
                "three": "/three.module.js",
                "three/addons/controls/OrbitControls.js": "/three-addons/controls/OrbitControls.js",
                "three/addons/loaders/GLTFLoader.js": "/three-addons/loaders/GLTFLoader.js",
                "three/addons/loaders/DRACOLoader.js": "/three-addons/loaders/DRACOLoader.js",
                "three/addons/libs/meshopt_decoder.module.js": "/three-addons/libs/meshopt_decoder.module.js",
                "three/addons/loaders/FBXLoader.js": "/three-addons/loaders/FBXLoader.js",
                "three/addons/environments/RoomEnvironment.js": "/three-addons/environments/RoomEnvironment.js",
                "three/addons/libs/stats.module.js": "/three-addons/libs/stats.module.js"
              }
            })
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
