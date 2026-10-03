/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.infra.embedding;

import com.nageoffer.ai.ragent.infra.enums.ModelProvider;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Local deterministic embedding for demos without external model credentials.
 */
@Service
public class NoopEmbeddingClient implements EmbeddingClient {

    private static final int DEFAULT_DIMENSION = 1536;

    @Override
    public String provider() {
        return ModelProvider.NOOP.getId();
    }

    @Override
    public List<Float> embed(String text, ModelTarget target) {
        int dimension = target != null
                && target.candidate() != null
                && target.candidate().getDimension() != null
                ? target.candidate().getDimension()
                : DEFAULT_DIMENSION;
        float[] vector = new float[dimension];
        String value = text == null ? "" : text;

        for (int i = 0; i < value.length(); i++) {
            char current = value.charAt(i);
            addFeature(vector, current, 1.0f);
            if (i + 1 < value.length()) {
                addFeature(vector, current * 31 + value.charAt(i + 1), 0.7f);
            }
        }

        normalize(vector);

        List<Float> result = new ArrayList<>(dimension);
        for (float v : vector) {
            result.add(v);
        }
        return result;
    }

    @Override
    public List<List<Float>> embedBatch(List<String> texts, ModelTarget target) {
        List<List<Float>> result = new ArrayList<>(texts.size());
        for (String text : texts) {
            result.add(embed(text, target));
        }
        return result;
    }

    private void addFeature(float[] vector, int value, float weight) {
        int index = Math.floorMod(value, vector.length);
        vector[index] += weight;
    }

    private void normalize(float[] vector) {
        double norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        if (norm == 0) {
            vector[0] = 1.0f;
            return;
        }
        float divisor = (float) Math.sqrt(norm);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= divisor;
        }
    }
}
