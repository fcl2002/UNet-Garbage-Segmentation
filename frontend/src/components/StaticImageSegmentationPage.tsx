// src/components/StaticImageSegmentationPage.tsx
import { useEffect, useRef, useState } from "react";

type SegmentationStats = {
  contains_trash: boolean | null;
  trash_pixel_ratio: number | null;
  mean_trash_probability: number | null;
};

// Se quiser depois, pode trocar para import.meta.env.VITE_API_URL
const API_URL = "http://localhost:8000/api/v1/predict/file";

type StatusType = "idle" | "loading" | "error" | "success";

export function StaticImageSegmentationPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<SegmentationStats | null>(null);
  const [status, setStatus] = useState<StatusType>("idle");
  const [statusText, setStatusText] = useState<string>("Prêt");

  // ---------- Status helpers ----------
  function updateStatus(text: string, type: StatusType) {
    setStatus(type);
    setStatusText(text);
  }

  function statusBadgeClass() {
    switch (status) {
      case "loading":
        return "bg-slate-800 text-yellow-300";
      case "error":
        return "bg-red-900 text-red-200";
      case "success":
        return "bg-slate-900 text-emerald-300";
      case "idle":
      default:
        return "bg-slate-900 text-indigo-300";
    }
  }

  // ---------- Manipulação de arquivos ----------
  function handleClickDropzone() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    handleImageFile(file);
  }

  function handleImageFile(file: File) {
    if (!file.type.startsWith("image/")) {
      alert("Veuillez sélectionner un fichier image.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      setPreviewUrl(String(ev.target?.result));
    };
    reader.readAsDataURL(file);

    void sendToApi(file);
  }

  // ---------- Chamada à API ----------
  async function sendToApi(file: File) {
    const formData = new FormData();
    formData.append("file", file);

    updateStatus("Traitement...", "loading");
    setStats(null);
    setResultImageUrl(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = "Erreur lors de l'appel à l'API";
        try {
          const data = await response.json();
          errorMessage = (data as any).detail || JSON.stringify(data);
        } catch {
          const text = await response.text();
          if (text) errorMessage = text;
        }
        throw new Error(errorMessage);
      }

      const containsTrash = response.headers.get("X-Contains-Trash");
      const trashRatio = response.headers.get("X-Trash-Ratio");
      const meanProb = response.headers.get("X-Mean-Trash-Prob");

      const parsedStats: SegmentationStats = {
        contains_trash:
          containsTrash === "True" ||
          containsTrash === "true" ||
          containsTrash === "1",
        trash_pixel_ratio: trashRatio ? parseFloat(trashRatio) : null,
        mean_trash_probability: meanProb ? parseFloat(meanProb) : null,
      };

      setStats(parsedStats);

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImageUrl(imageUrl);

      updateStatus("Succès", "success");
    } catch (err: any) {
      console.error(err);
      setStats(null);
      setResultImageUrl(null);
      updateStatus("Erreur", "error");
    }
  }

  // ---------- Handler de Ctrl+V (colar imagem) ----------
  useEffect(() => {
    function handlePaste(event: ClipboardEvent) {
      const items = event.clipboardData?.items;
      if (!items) return;

      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            handleImageFile(file);
            event.preventDefault();
          }
          break;
        }
      }
    }

    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
<div className="w-full md:w-[1200px] mx-auto bg-slate-900/80 border border-slate-700 rounded-3xl shadow-[0_18px_40px_rgba(0,0,0,0.6)] px-4 py-6 md:px-8 md:py-8 flex flex-col gap-6">
      {/* Seção: imagem de entrada */}
      <section className="flex flex-col">
        <h2 className="text-2xl font-semibold mb-1 text-slate-50">Image</h2>
        <p className="text-slate-300 text-sm md:text-base mb-3">
          Collez une image ou importez un fichier <code>.jpg</code> ou{" "}
          <code>.png</code>.
        </p>

        <div
          className={`relative min-h-[260px] max-h-[420px] rounded-2xl border-2 border-dashed transition-colors duration-200 ${
            previewUrl
              ? "border-emerald-400 bg-slate-950"
              : "border-slate-600 bg-slate-900 hover:border-emerald-400"
          } cursor-pointer flex flex-col`}
          onClick={handleClickDropzone}
        >
          {!previewUrl && (
            <div className="flex flex-1 items-center justify-center px-6 py-6 text-center">
              <div>
                <div className="text-base md:text-lg font-medium mb-4 text-slate-100">
                  Cliquez pour sélectionner un fichier ou collez une image
                  depuis le presse-papier.
                </div>
                <button
                  type="button"
                  className="inline-flex items-center justify-center px-5 py-3 rounded-full bg-emerald-500 text-slate-950 text-sm md:text-base font-semibold shadow-md hover:bg-emerald-400 hover:shadow-lg transition"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClickDropzone();
                  }}
                >
                  <span className="mr-2">📁</span>
                  <span>Choisir un fichier</span>
                </button>
                <div className="mt-3 text-xs md:text-sm text-slate-400">
                  Astuce : copiez une image puis appuyez sur{" "}
                  <strong>Ctrl+V</strong> ici.
                </div>
              </div>
            </div>
          )}

          {previewUrl && (
            <div className="w-full h-full rounded-2xl overflow-hidden">
              <img
                src={previewUrl}
                alt="Pré-visualisation"
                className="w-full h-full object-cover"
              />
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>
      </section>

      {/* Seção: resultado */}
      <section className="flex flex-col">
        <h2 className="text-2xl font-semibold mb-2 text-slate-50">Résultat</h2>

        <div className="flex flex-col rounded-2xl bg-slate-950/90 text-slate-100 p-4 border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-slate-100">
              Sortie de l&apos;API
            </h3>
            <div
              className={`text-xs px-3 py-1 rounded-full font-medium ${statusBadgeClass()}`}
            >
              {statusText}
            </div>
          </div>

          <div className="mt-2 mb-3 rounded-xl bg-slate-900 flex items-center justify-center min-h-[200px] max-h-[360px] overflow-hidden">
            {resultImageUrl ? (
              <img
                src={resultImageUrl}
                alt="Image segmentée"
                className="w-full h-full object-contain"
              />
            ) : (
              <span className="text-xs text-slate-500">
                Envoyez une image pour voir ici la version segmentée.
              </span>
            )}
          </div>

          <pre className="flex-1 text-xs leading-snug whitespace-pre-wrap break-words overflow-y-auto bg-slate-900 rounded-xl px-3 py-2 text-slate-200 max-h-52">
            {stats
              ? JSON.stringify(stats, null, 2)
              : "Envoyez une image pour voir ici les statistiques de la segmentation."}
          </pre>

          <p className="mt-2 text-xs text-slate-400">
            L&apos;API appelée est{" "}
            <code className="bg-slate-800 px-1 rounded">
              POST /api/v1/predict/file
            </code>
            . L&apos;image retournée est la version segmentée.
          </p>
        </div>
      </section>
    </div>
  );
}
