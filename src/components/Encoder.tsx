import { useState } from 'react';
import { loadImage, imageToTensor, tensorToImage, stringToBits, padBits, canvasToBlob } from '../utils/imageProcessing';
import { useStegaStamp } from '../hooks/useStegaStamp';

const ENCODER_WIDTH = 224;
const ENCODER_HEIGHT = 224;
const SECRET_BITS = 256;
const ECC_BITS = 0;

export function Encoder({ useWebGPU = false }: { useWebGPU?: boolean }) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [secret, setSecret] = useState('');
  const [output, setOutput] = useState<HTMLCanvasElement | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');

  const { model, loading, error } = useStegaStamp({
    encoderPath: '/models/encoder.onnx',
    decoderPath: '/models/decoder.onnx',
    useWebGPU
  });

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const img = await loadImage(file);
      setImage(img);
      setStatus('Image loaded');
    } catch (err) {
      setStatus(`Error loading image: ${err}`);
    }
  };

  const handleEncode = async () => {
    if (!image || !model || !secret) {
      setStatus('Please upload an image and enter a secret');
      return;
    }

    setIsProcessing(true);
    setStatus('Encoding...');

    try {
      const { data: imageTensor, dims } = imageToTensor(image, ENCODER_WIDTH, ENCODER_HEIGHT);

      let secretBits = stringToBits(secret);
      secretBits = padBits(secretBits, SECRET_BITS);

      const secretBitValues = new Uint8Array(secretBits.length);
      for (let i = 0; i < secretBits.length; i++) {
        secretBitValues[i] = secretBits[i] ? 1 : 0;
      }

      const encoded = await model.encode(imageTensor, secretBitValues, dims);

      const encodedImage = tensorToImage(encoded, ENCODER_WIDTH, ENCODER_HEIGHT);
      const canvas = document.createElement('canvas');
      canvas.width = ENCODER_WIDTH;
      canvas.height = ENCODER_HEIGHT;
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(encodedImage, 0, 0);

      setOutput(canvas);
      setStatus('Encoding complete!');
    } catch (err) {
      setStatus(`Error during encoding: ${err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = async () => {
    if (!output) return;
    const blob = await canvasToBlob(output);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `stegastamp_${Date.now()}.png`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="panel">Loading model...</div>;
  if (error) return <div className="panel" style={{ color: 'red' }}>Error: {error}</div>;

  return (
    <div className="panel">
      <h2>StegaStamp Encoder</h2>

      <div className="form-group">
        <label>Upload Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
        />
        {image && (
          <div style={{ marginTop: '0.5rem' }}>
            <img
              src={image.src}
              alt="Uploaded"
              style={{ maxWidth: '20rem', height: 'auto', border: '1px solid #ddd', borderRadius: '0.25rem' }}
            />
          </div>
        )}
      </div>

      <div className="form-group">
        <label>Secret (max 32 bytes)</label>
        <input
          type="text"
          maxLength={32}
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          placeholder="Enter secret message..."
        />
      </div>

      <button
        onClick={handleEncode}
        disabled={isProcessing || !image || !secret}
      >
        {isProcessing ? 'Encoding...' : 'Encode'}
      </button>

      {status && <div className="status">{status}</div>}

      {output && (
        <div className="preview-container">
          <h3>Encoded Output</h3>
          <canvas
            ref={(canvas) => {
              if (canvas && output) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                  ctx.drawImage(output, 0, 0);
                }
              }
            }}
            width={ENCODER_WIDTH}
            height={ENCODER_HEIGHT}
            style={{ border: '1px solid #ddd' }}
          />
          <button
            onClick={handleDownload}
            style={{ marginTop: '0.5rem' }}
          >
            Download Encoded Image
          </button>
        </div>
      )}
    </div>
  );
}
