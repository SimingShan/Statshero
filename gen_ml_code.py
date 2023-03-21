def gen_function(data, x, y):
    model = LinearRegression()
    model.fit(data[x], data[y])
    predictions = model.predict(data[x])
    plt.scatter(data[x], data[y])
    plt.plot(data[x], predictions)
    return model.coef_, model.intercept_