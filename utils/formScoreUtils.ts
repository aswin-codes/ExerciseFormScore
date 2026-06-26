

export type Point2D = { x: number; y: number; score: number };

export type FramePose = {
  tSec: number;
  keypoints: Float32Array; // 51 floats: 17 keypoints * (y, x, score), MoveNet layout
};

export type Side = 'left' | 'right';

// ---- MoveNet single-pose keypoint indices ----
export const KP = {
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  RIGHT_ELBOW: 8,
  LEFT_WRIST: 9,
  RIGHT_WRIST: 10,
  LEFT_HIP: 11,
  RIGHT_HIP: 12,
} as const;

export const MIN_CONFIDENCE = 0.3;

function kpAt(keypoints: Float32Array, index: number): Point2D {
  return {
    y: keypoints[index * 3],
    x: keypoints[index * 3 + 1],
    score: keypoints[index * 3 + 2],
  } as Point2D;
}

export function calculateJointAngle(a: Point2D, b: Point2D, c: Point2D): number {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

// ---- Exercise configuration ----
// Every exercise is just a list of joints to track. Add a new exercise by
// adding a new ExerciseConfig — nothing else in this file needs to change.

export type JointDef = {
  name: string; // stable key, e.g. 'elbow'
  label: string; // human-readable, e.g. 'Elbow flexion'
  weight: number; // contribution to overall score (normalized automatically)
  toleranceDeg: number; // degrees of difference still considered "good"
  left: [number, number, number]; // [pointA, vertex, pointC] keypoint indices, left side
  right: [number, number, number]; // same triple, right side
};

export type ExerciseConfig = {
  id: string;
  label: string;
  joints: JointDef[];
};

export const BICEP_CURL_CONFIG: ExerciseConfig = {
  id: 'bicep_curl',
  label: 'Bicep curl',
  joints: [
    {
      // The actual curl motion: shoulder-elbow-wrist flexion angle.
      name: 'elbow',
      label: 'Elbow flexion',
      weight: 0.7,
      toleranceDeg: 15,
      left: [KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST],
      right: [KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST],
    },
    {
      // Upper-arm-to-torso angle. Should stay roughly constant through a clean
      // curl — a rising/falling value here is the classic "swinging the
      // shoulder for momentum" cheat.
      name: 'shoulder_stability',
      label: 'Shoulder stability (no swinging)',
      weight: 0.3,
      toleranceDeg: 12,
      left: [KP.LEFT_HIP, KP.LEFT_SHOULDER, KP.LEFT_ELBOW],
      right: [KP.RIGHT_HIP, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW],
    },
  ],
};

// ---- Per-frame angle extraction ----

export type AngleSeries = {
  side: Side;
  tSec: number[];
  angles: Record<string, number[]>; // angles[jointName][frameIdx]
  confidences: Record<string, number[]>; // confidences[jointName][frameIdx]
};

function pickDominantSide(frames: FramePose[], joint: JointDef): Side {
  let leftTotal = 0;
  let rightTotal = 0;
  for (const frame of frames) {
    leftTotal += kpAt(frame.keypoints, joint.left[1]).score;
    rightTotal += kpAt(frame.keypoints, joint.right[1]).score;
  }
  return leftTotal >= rightTotal ? 'left' : 'right';
}

// Picks one side (left or right) for the WHOLE video based on which arm was
// more confidently tracked overall, then uses that same side for every joint.
// Switching sides frame-by-frame (as a naive implementation might) causes
// visible jumps in the angle signal whenever confidence flips momentarily.
export function computeAngleSeries(frames: FramePose[], config: ExerciseConfig): AngleSeries {
  const side = pickDominantSide(frames, config.joints[0]);

  const tSec: number[] = [];
  const angles: Record<string, number[]> = {};
  const confidences: Record<string, number[]> = {};
  for (const joint of config.joints) {
    angles[joint.name] = [];
    confidences[joint.name] = [];
  }

  for (const frame of frames) {
    tSec.push(frame.tSec);
    for (const joint of config.joints) {
      const [aIdx, bIdx, cIdx] = side === 'left' ? joint.left : joint.right;
      const a = kpAt(frame.keypoints, aIdx);
      const b = kpAt(frame.keypoints, bIdx);
      const c = kpAt(frame.keypoints, cIdx);
      angles[joint.name].push(calculateJointAngle(a, b, c));
      confidences[joint.name].push(Math.min(a.score, b.score, c.score));
    }
  }

  // --- Interpolate low-confidence gaps ---
  for (const joint of config.joints) {
    const jointAngles = angles[joint.name];
    const jointConf = confidences[joint.name];
    
    let lastValidIdx = -1;
    for (let i = 0; i < jointAngles.length; i++) {
      if (jointConf[i] >= MIN_CONFIDENCE) {
        if (lastValidIdx !== -1 && i - lastValidIdx > 1) {
          // Linearly interpolate between lastValidIdx and i
          const startVal = jointAngles[lastValidIdx];
          const endVal = jointAngles[i];
          const steps = i - lastValidIdx;
          for (let j = 1; j < steps; j++) {
            jointAngles[lastValidIdx + j] = startVal + (endVal - startVal) * (j / steps);
            jointConf[lastValidIdx + j] = MIN_CONFIDENCE; // Boost conf so it's valid
          }
        } else if (lastValidIdx === -1 && i > 0) {
          // If array starts with low confidence, backfill with the first valid value
          for (let j = 0; j < i; j++) {
            jointAngles[j] = jointAngles[i];
            jointConf[j] = MIN_CONFIDENCE;
          }
        }
        lastValidIdx = i;
      }
    }
    
    // If array ends with low confidence, forward-fill with the last valid value
    if (lastValidIdx !== -1 && lastValidIdx < jointAngles.length - 1) {
      for (let j = lastValidIdx + 1; j < jointAngles.length; j++) {
        jointAngles[j] = jointAngles[lastValidIdx];
        jointConf[j] = MIN_CONFIDENCE;
      }
    }
  }

  return { side, tSec, angles, confidences };
}

export function pearsonCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
  
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const meanX = sumX / n;
  const meanY = sumY / n;
  
  let num = 0;
  let denX = 0;
  let denY = 0;
  
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  
  const den = Math.sqrt(denX * denY);
  if (den === 0) return 0;
  return num / den;
}

// ---- DTW alignment ----
// Aligns two angle sequences of possibly different length/speed. Cost between
// a user frame and an expert frame is a confidence-weighted, tolerance-
// normalized distance across all tracked joints — so a momentarily occluded
// joint doesn't drag the alignment off course, and a joint with a tight
// tolerance (like the elbow) matters more than a loose one.
function frameCost(
  config: ExerciseConfig,
  userAngles: AngleSeries,
  expertAngles: AngleSeries,
  i: number,
  j: number,
): number {
  let weightedDiff = 0;
  let totalWeight = 0;

  for (const joint of config.joints) {
    const uAngle = userAngles.angles[joint.name][i];
    const eAngle = expertAngles.angles[joint.name][j];
    const normDiff = Math.abs(uAngle - eAngle) / joint.toleranceDeg;

    weightedDiff += joint.weight * normDiff;
    totalWeight += joint.weight;
  }

  if (totalWeight === 0) return 1;
  return weightedDiff / totalWeight;
}

export function runDTW(
  config: ExerciseConfig,
  userAngles: AngleSeries,
  expertAngles: AngleSeries,
): Array<[number, number]> {
  const n = userAngles.tSec.length;
  const m = expertAngles.tSec.length;
  if (n === 0 || m === 0) return [];

  const D: number[][] = Array.from({ length: n }, () => new Array(m).fill(Infinity));
  const prev: Array<Array<[number, number] | null>> = Array.from({ length: n }, () =>
    new Array(m).fill(null),
  );

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const c = frameCost(config, userAngles, expertAngles, i, j);

      if (i === 0 && j === 0) {
        D[i][j] = c;
        continue;
      }

      const candidates: Array<[number, [number, number]]> = [];
      if (i > 0) candidates.push([D[i - 1][j], [i - 1, j]]);
      if (j > 0) candidates.push([D[i][j - 1], [i, j - 1]]);
      if (i > 0 && j > 0) candidates.push([D[i - 1][j - 1], [i - 1, j - 1]]);

      candidates.sort((a, b) => a[0] - b[0]);
      D[i][j] = c + candidates[0][0];
      prev[i][j] = candidates[0][1];
    }
  }

  const path: Array<[number, number]> = [];
  let cur: [number, number] | null = [n - 1, m - 1];
  while (cur) {
    path.push(cur);
    cur = prev[cur[0]][cur[1]];
  }
  path.reverse();
  return path;
}

// ---- Scoring ----

export function scoreFromPath(
  config: ExerciseConfig,
  userAngles: AngleSeries,
  expertAngles: AngleSeries,
  path: Array<[number, number]>,
): { overallScore: number; jointScores: Record<string, number> } {
  const jointScores: Record<string, number> = {};
  let overallWeighted = 0;
  let overallWeight = 0;

  for (const joint of config.joints) {
    const alignedUserAngles: number[] = [];
    const alignedExpertAngles: number[] = [];
    
    let sumNormDiff = 0;
    
    for (const [i, j] of path) {
      const uAngle = userAngles.angles[joint.name][i];
      const eAngle = expertAngles.angles[joint.name][j];
      
      alignedUserAngles.push(uAngle);
      alignedExpertAngles.push(eAngle);
      
      const normDiff = Math.abs(uAngle - eAngle) / joint.toleranceDeg;
      sumNormDiff += normDiff;
    }
    
    const pathLength = path.length;
    if (pathLength === 0) {
      jointScores[joint.name] = 0;
      continue;
    }
    
    // 1. Range of Motion Score (Absolute Difference)
    const meanNormDiff = sumNormDiff / pathLength;
    let romScore = 100 * Math.max(0, 1 - meanNormDiff);
    
    // 2. Shape Score (Pearson Correlation)
    let shapeScore = 0;
    const r = pearsonCorrelation(alignedUserAngles, alignedExpertAngles);
    if (r > 0) {
      shapeScore = r * 100;
    }
    
    // Blend 50/50 for the final score
    const finalJointScore = Math.round((romScore * 0.5) + (shapeScore * 0.5));
    
    jointScores[joint.name] = finalJointScore;
    overallWeighted += joint.weight * finalJointScore;
    overallWeight += joint.weight;
  }

  const overallScore = overallWeight > 0 ? Math.round(overallWeighted / overallWeight) : 0;
  return { overallScore, jointScores };
}

// ---- Chart data ----

export type AlignedChartSeries = {
  labels: string[];
  userValues: number[];
  expertValues: number[];
};

export function buildAlignedChartSeries(
  userAngles: AngleSeries,
  expertAngles: AngleSeries,
  path: Array<[number, number]>,
  jointName: string,
): AlignedChartSeries {
  const userValues = path.map(([i]) => Math.round(userAngles.angles[jointName][i]));
  const expertValues = path.map(([, j]) => Math.round(expertAngles.angles[jointName][j]));

  const totalSteps = path.length;
  const labelEvery = Math.max(1, Math.ceil(totalSteps / 6));
  const labels = path.map((_, idx) =>
    idx % labelEvery === 0
      ? `${Math.round((idx / Math.max(1, totalSteps - 1)) * 100)}%`
      : '',
  );

  return { labels, userValues, expertValues };
}

// ---- Top-level entry point ----

export type FormScoreResult = {
  overallScore: number;
  jointScores: Record<string, number>;
  chart: AlignedChartSeries;
  userSide: Side;
  expertSide: Side;
};

export function computeFormScore(
  userFrames: FramePose[],
  expertFrames: FramePose[],
  config: ExerciseConfig,
): FormScoreResult {
  const userAngles = computeAngleSeries(userFrames, config);
  const expertAngles = computeAngleSeries(expertFrames, config);

  console.log('--- FORM SCORE LOGS ---');
  console.log('User Angles:', JSON.stringify(userAngles.angles, null, 2));
  console.log('Expert Angles:', JSON.stringify(expertAngles.angles, null, 2));

  const path = runDTW(config, userAngles, expertAngles);
  const { overallScore, jointScores } = scoreFromPath(config, userAngles, expertAngles, path);
  const chart = buildAlignedChartSeries(userAngles, expertAngles, path, config.joints[0].name);

  console.log('Overall Score:', overallScore);
  console.log('Joint Scores:', JSON.stringify(jointScores, null, 2));
  console.log('-----------------------');

  return {
    overallScore,
    jointScores,
    chart,
    userSide: userAngles.side,
    expertSide: expertAngles.side,
  };
}