import { useEffect, useState, useRef, useMemo } from 'react';
import StegaStampModel, { ModelConfig } from '../models/StegaStampModel';

export function useStegaStamp(config: ModelConfig) {
  const [model, setModel] = useState<StegaStampModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<StegaStampModel | null>(null);

  const memoConfig = useMemo(() => config, [config.encoderPath, config.decoderPath, config.useWebGPU]);

  useEffect(() => {
    const initModel = async () => {
      try {
        const instance = new StegaStampModel(memoConfig);
        await instance.init();
        modelRef.current = instance;
        setModel(instance);
        setLoading(false);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load model';
        setError(message);
        setLoading(false);
      }
    };

    initModel();

    return () => {
      modelRef.current?.dispose();
    };
  }, [memoConfig]);

  return { model, loading, error };
}
