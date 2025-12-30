from sklearn.linear_model import LinearRegression

def predict_next_year(df):
    predictions = {}

    states = df['state'].unique()

    for state in states:
        state_data = df[df['state'] == state]

        yearly_state = state_data.groupby('year')[['loss', 'fatalities']].sum().reset_index()

        if len(yearly_state) < 3:
            predictions[state] = {"loss": 0, "fatalities": 0}
            continue

        X = yearly_state['year'].values.reshape(-1, 1)

        y_loss = yearly_state['loss'].values
        model_loss = LinearRegression()
        model_loss.fit(X, y_loss)
        pred_loss = model_loss.predict([[2025]])[0]

        y_fat = yearly_state['fatalities'].values
        model_fat = LinearRegression()
        model_fat.fit(X, y_fat)
        pred_fat = model_fat.predict([[2025]])[0]

        predictions[state] = {
            "loss": float(max(0, pred_loss)),  # No negative loss
            "fatalities": int(max(0, pred_fat))  # No negative fatalities
        }

    return predictions
