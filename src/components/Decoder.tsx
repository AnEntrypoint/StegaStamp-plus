import { useState } from 'react';
import { loadImage, imageToTensor, bytesToBits, bitsToString } from '../utils/imageProcessing';
import { useStegaStamp } from '../hooks/useStegaStamp';

const DECODER_WIDTH = 224;
const DECODER_HEIGHT = 224;
const SECRET_BITS = 256;

export function Decoder({ useWebGPU = false }: { useWebGPU?: boolean }) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [decoded, setDecoded] = useState('');
  const [confidence, setConfidence] = useState(0);
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

  const handleDecode = async () => {
    if (!image || !model) {
      setStatus('Please upload an image');
      return;
    }

    setIsProcessing(true);
    setStatus('Decoding...');

    try {
      const { data: imageTensor, dims } = imageToTensor(image, DECODER_WIDTH, DECODER_HEIGHT);
      const result = await model.decode(imageTensor, dims);

      const corrected = result.bits;

      const bits = bytesToBits(corrected);
      let secret = '';
      try {
        secret = bitsToString(bits.slice(0, SECRET_BITS)).split('\0')[0];
      } catch {
        secret = '(unable to decode)';
      }

      setDecoded(secret);
      setConfidence(result.confidence);
      setStatus('Decoding complete!');
    } catch (err) {
      setStatus(`Error during decoding: ${err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  if (loading) return <div className="panel">Loading model...</div>;
  if (error) return <div className="panel" style={{ color: 'red' }}>Error: {error}</div>;

  return (
    <div className="panel">
      <h2>StegaStamp Decoder</h2>

      <div className="form-group">
        <label>Upload Encoded Image</label>
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

      <button
        onClick={handleDecode}
        disabled={isProcessing || !image}
      >
        {isProcessing ? 'Decoding...' : 'Decode'}
      </button>

      {status && <div className="status">{status}</div>}

      {decoded && (
        <div className="result-box">
          <h3>Decoded Secret</h3>
          <div className="decoded-text">{decoded}</div>
          <div className="confidence">
            Confidence: {(confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
