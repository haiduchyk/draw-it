namespace FreeDraw
{
    using System.Collections.Generic;
    using UnityEngine;

    public class Draw : MonoBehaviour
    {
        [SerializeField]
        private List<Color> colors;

        [SerializeField]
        private int penWidth = 3;

        [SerializeField]
        private LayerMask drawingLayers;

        [SerializeField]
        private bool resetCanvasOnPlay = true;

        [SerializeField]
        private Color resetColour = new Color(0, 0, 0, 0); // By default, reset the canvas to be transparent

        private Color PenColour => colors[Random.Range(0, colors.Count)];

        private Texture2D drawableTexture;
        private Color[] cleanColoursArray;
        private Sprite drawableSprite;
        private Vector2 previousDragPosition;
        private Color transparent;
        private Color32[] curColors;
        private bool mouseWasPreviouslyHeldDown;
        private bool noDrawingONCurrentDrag;

        private void Awake()
        {
            drawableSprite = GetComponent<SpriteRenderer>().sprite;
            drawableTexture = drawableSprite.texture;

            cleanColoursArray = new Color[(int) drawableSprite.rect.width * (int) drawableSprite.rect.height];

            for (var x = 0; x < cleanColoursArray.Length; x++)
            {
                cleanColoursArray[x] = resetColour;
            }

            if (resetCanvasOnPlay)
            {
                ResetCanvas();
            }
        }

        [EditorButton]
        public void ResetCanvas()
        {
            drawableTexture.SetPixels(cleanColoursArray);
            drawableTexture.Apply();
        }

        private void Update()
        {
            var mouseHeldDown = Input.GetMouseButton(0);
            if (mouseHeldDown && !noDrawingONCurrentDrag)
            {
                Vector2 mouseWorldPosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);

                var hit = Physics2D.OverlapPoint(mouseWorldPosition, drawingLayers.value);
                if (hit != null)
                {
                    PenBrush(mouseWorldPosition);
                }
                else
                {
                    previousDragPosition = Vector2.zero;
                    if (!mouseWasPreviouslyHeldDown)
                    {
                        noDrawingONCurrentDrag = true;
                    }
                }
            }
            else if (!mouseHeldDown)
            {
                previousDragPosition = Vector2.zero;
                noDrawingONCurrentDrag = false;
            }

            mouseWasPreviouslyHeldDown = mouseHeldDown;
        }

        private void PenBrush(Vector2 worldPoint)
        {
            var pixelPos = WorldToPixelCoordinates(worldPoint);

            curColors = drawableTexture.GetPixels32();

            if (previousDragPosition == Vector2.zero)
            {
                MarkPixelsToColour(pixelPos, penWidth, PenColour);
            }
            else
            {
                ColourBetween(previousDragPosition, pixelPos, penWidth, PenColour);
            }

            ApplyMarkedPixelChanges();

            previousDragPosition = pixelPos;
        }


        private void ColourBetween(Vector2 startPoint, Vector2 endPoint, int width, Color color)
        {
            var distance = Vector2.Distance(startPoint, endPoint);
            var lerpSteps = 1 / distance;

            for (float lerp = 0; lerp <= 1; lerp += lerpSteps)
            {
                var curPosition = Vector2.Lerp(startPoint, endPoint, lerp);
                MarkPixelsToColour(curPosition, width, color);
            }
        }

        private void MarkPixelsToColour(Vector2 centerPixel, int penThickness, Color colorOfPen)
        {
            var centerX = (int) centerPixel.x;
            var centerY = (int) centerPixel.y;

            for (var x = centerX - penThickness; x <= centerX + penThickness; x++)
            {
                if (x >= (int) drawableSprite.rect.width || x < 0)
                {
                    continue;
                }

                for (var y = centerY - penThickness; y <= centerY + penThickness; y++)
                {
                    MarkPixelToChange(x, y, colorOfPen);
                }
            }
        }

        private void MarkPixelToChange(int x, int y, Color color)
        {
            var arrayPos = y * (int) drawableSprite.rect.width + x;

            if (arrayPos > curColors.Length || arrayPos < 0)
            {
                return;
            }

            curColors[arrayPos] = color;
        }

        private Vector2 WorldToPixelCoordinates(Vector2 worldPosition)
        {
            var localPos = transform.InverseTransformPoint(worldPosition);

            var pixelWidth = drawableSprite.rect.width;
            var pixelHeight = drawableSprite.rect.height;
            var unitsToPixels = pixelWidth / drawableSprite.bounds.size.x * transform.localScale.x;

            var centeredX = localPos.x * unitsToPixels + pixelWidth / 2;
            var centeredY = localPos.y * unitsToPixels + pixelHeight / 2;

            var pixelPos = new Vector2(Mathf.RoundToInt(centeredX), Mathf.RoundToInt(centeredY));

            return pixelPos;
        }

        private void ApplyMarkedPixelChanges()
        {
            drawableTexture.SetPixels32(curColors);
            drawableTexture.Apply();
        }
    }
}